import random
import torch

import torch.nn.functional as F
from typing import List
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
from torch.nn import CrossEntropyLoss

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from torch.utils.data import TensorDataset, DataLoader

def make_ranker(ranker_name:str, model_path=None, device=None, *args):
    if ranker_name == 'random':
        return RandomPruner(*args)
    elif ranker_name == 'loss':
        return LossPruner(model_path=model_path, device=device)
    elif ranker_name == 'kmeans':
        return KMeansPruner(model_path=model_path, device=device)
    else:
        raise ValueError("invalid ranking option")

### Super DataPruner Classes #########################################################
class DataPruner():
    """ base class for all rankers """
    def __init__(self, seed_num:int=None):
        if seed_num:
            random.seed(seed_num)
    
    def rank_data(self, data:List)->List:
        data_copy = deepcopy(data)
        data_copy = sorted(data_copy, key=lambda x: self.get_ex_score(x), reverse=True)
        return data_copy
    
    def get_ex_score(self, ex:SimpleNamespace)->float:
        pass
    
    def filter_data(self, data:List, ret_frac:float)->List:
        N = int(ret_frac*len(data))
        data = self.rank_data(data)
        return data[:N]   

    def __call__(self, *args, **kwargs):
        return self.filter_data(*args, **kwargs)

class ModelDataPruner(DataPruner):
    def __init__(self, seed_num:int=None, model_path=None, device=None):
        super().__init__(seed_num)
        if model_path:
            self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.device = device
    
    def filter_data(self, *args, **kwargs)->List:
        """ overwriting parent to free model gpu memory after use """
        output = super().filter_data(*args, **kwargs)
        self.model.to('cpu')
        torch.cuda.empty_cache() 
        return output
    
    @staticmethod
    def tensor_prep(ex:dict):
        label = torch.LongTensor([ex['output']])
        inp_id = torch.LongTensor([ex['inputs'][0]])
        t_id = torch.LongTensor([ex['inputs'][1]])
        attn_m = torch.LongTensor([ex['inputs'][2]])
        return inp_id, t_id, attn_m, label
    
    @staticmethod
    def dl_prep(labels:list, input_ids:list, token_type_ids:list, attention_masks:list, bs=32):
        labels = torch.from_numpy(np.array(labels))
        labels = labels.long()
        input_ids = torch.from_numpy(np.array(input_ids))
        input_ids = input_ids.long()
        token_type_ids = torch.from_numpy(np.array(token_type_ids))
        token_type_ids = token_type_ids.long()
        attention_masks = torch.from_numpy(np.array(attention_masks))
        attention_masks = attention_masks.long()
    
        ds = TensorDataset(input_ids, token_type_ids, attention_masks, labels)
        dl = DataLoader(ds, batch_size=bs, shuffle=False)
        return dl

        
    
### Basic Rankers ###############################################################

class RandomPruner(DataPruner):
    """ ranks all examples in a random order based on the seed """
    def __init__(self, seed_num:int=None):
        super().__init__(seed_num)
    def get_ex_score(self)->float:
        return random.random()

### Model Based Rankers #########################################################

class LossPruner(ModelDataPruner):
    """ ranks all examples based on the loss of a model trained already on the examples """
    def __init__(self, seed_num:int=1, model_path=None, reverse=False, device=None):
        super().__init__(seed_num, model_path=model_path, device=device)
        self.reverse = reverse
        self.counter = 0
    
    def filter_data(self, data:List, ret_frac:float, batch_size=1) -> List:
        if not self.device:
            return super().filter_data(data=data, ret_frac=ret_frac)

        N = int(ret_frac*len(data))
        labels = [d['output'] for d in data]
        input_ids = [d['inputs'][0] for d in data]
        token_type_ids = [d['inputs'][1] for d in data]
        attention_masks = [d['inputs'][2] for d in data]
        
        dl = self.dl_prep(labels, input_ids, token_type_ids, attention_masks, bs=batch_size) 

        self.model.to(self.device)
        self.model.eval()


        # all_logits = []
        # for i, (inp_id, tok_typ_id, att_msk, lab) in enumerate(dl):
        #     print(f'On {i}/{len(dl)}')
        #     inp_id, tok_typ_id, att_msk = inp_id.to(self.device), tok_typ_id.to(self.device), att_msk.to(self.device)
        #     with torch.no_grad():
        #         outputs = self.model(input_ids=inp_id, attention_mask=att_msk, token_type_ids=tok_typ_id)
        #         logits = outputs[0].detach().cpu()
        #     all_logits.append(logits)
        #     print("Got curr logits")
        # logits = torch.cat(all_logits)
        # print("Getting losses now")

        # # Get losses
        # loss_fct = CrossEntropyLoss()
        # all_losses = []
        # for i in range(logits.size(0)):
        #     logg = logits[i,:]
        #     lab = torch.LongTensor(labels[i]).unsqueeze(dim=0)
        #     all_losses.append(loss_fct(logg, lab))

        all_losses = []
        for i, (inp_id, tok_typ_id, att_msk, lab) in enumerate(dl):
            print(f'On {i}/{len(dl)}')
            inp_id, tok_typ_id, att_msk = inp_id.to(self.device), tok_typ_id.to(self.device), att_msk.to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=inp_id, attention_mask=att_msk, token_type_ids=tok_typ_id, labels=lab)
                loss = outputs[0].item()
            all_losses.append(loss)

        losses=torch.cat(all_losses)
        inds = torch.argsort(losses, descending=not self.reverse).tolist()
        return [data[ind] for ind in inds]
     
    def get_ex_score(self, ex)->float:
        self.counter += 1
        print(self.counter)
        inp_id, t_id, attn_m, label = self.tensor_prep(ex)
        outputs = self.model(input_ids=inp_id, attention_mask=attn_m, token_type_ids=t_id, labels=label)
        loss = outputs[0].item()
        if self.reverse:
            loss = -1.*loss
        return float(loss)
        

class KMeansPruner(ModelDataPruner):
    '''
        View samples in encoder embedding space
        PCA compression of samples
        K-Means clustering
        Select K samples (closest to each cluster mean)
    '''
    def __init__(self, seed_num:int=1, model_path=None, device=None):
        super().__init__(seed_num, model_path=model_path, device=device)

     
    def filter_data(self, data:List, ret_frac:float, ncomps:int=10)->List:
        N = int(ret_frac*len(data))

        # Encoder embedding space
        if self.device:
            H = self.get_hidden_vecs_batched(data)
        else:
            H = self.get_hidden_vecs(data)

        print("size", H.size())
        # PCA compression
        pca = PCA(n_components=ncomps)
        compressed = pca.fit_transform(H)

        # K means clustering
        kmeans = KMeans(n_clusters=N, random_state=0).fit(compressed)
        
        # Select closest to mean per cluster
        cluster_means = kmeans.cluster_centers_
        selected_inds = []
        for k in range(N):
            mean_vector = cluster_means[k]
            diff_l2 = np.linalg.norm(compressed - mean_vector, axis=1)
            ind = np.argmin(diff_l2)
            selected_inds.append(ind)
        return [data[ind] for ind in selected_inds]
        

    def get_hidden_vecs(self, data):
        H = [self.get_hidden_vec(ex) for ex in data]
        H = torch.FloatTensor(H)
        return H
    
    def get_hidden_vec(self, ex)->List:
        '''
        Get hidden vec emb of correct label
        '''
        label_ind = int(ex['output'])
        inp_id = torch.LongTensor([ex['inputs'][0][label_ind]])
        tok_typ_id = torch.LongTensor([ex['inputs'][1][label_ind]])
        att_msk = torch.LongTensor([ex['inputs'][2][label_ind]])

        emb = self.model.electra(input_ids=inp_id, attention_mask=att_msk, token_type_ids=tok_typ_id)[0]
        emb_cls = emb[:,0,:]
        return emb_cls.squeeze().detach.cpu()
    
    def get_hidden_vecs_batched(self, data, batch_size=32):
        labels = [d['output'] for d in data]
        input_ids = [d['inputs'][0][l_ind] for d,l_ind in zip(data, labels)]
        token_type_ids = [d['inputs'][1][l_ind] for d,l_ind in zip(data, labels)]
        attention_masks = [d['inputs'][2][l_ind] for d,l_ind in zip(data, labels)]
        
        dl = self.dl_prep(labels, input_ids, token_type_ids, attention_masks, bs=batch_size) 

        self.model.to(self.device)
        self.model.eval()
        all_embs = []
        for i, (inp_id, tok_typ_id, att_msk, _) in enumerate(dl):
            # print(f'On {i}/{len(dl)}')
            inp_id, tok_typ_id, att_msk = inp_id.to(self.device), tok_typ_id.to(self.device), att_msk.to(self.device)
            with torch.no_grad():
                embs = self.model.electra(input_ids=inp_id, attention_mask=att_msk, token_type_ids=tok_typ_id)[0]
                embs_cls = embs[:,0,:]
            all_embs.append(embs_cls.detach().cpu())
        H = torch.cat(all_embs)
        return H
    


