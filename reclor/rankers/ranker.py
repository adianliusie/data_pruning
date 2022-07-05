import random
import torch

import torch.nn.functional as F
from typing import List
from types import SimpleNamespace
from copy import deepcopy
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def make_ranker(ranker_name:str, model=None, *args):
    if ranker_name == 'random':
        return RandomPruner(*args)
    elif ranker_name == 'loss':
        return LossPruner(model)
    elif ranker_name == 'kmeans':
        return KMeansPruner(model)
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
    def __init__(self, seed_num:int=None, model=None):
        super().__init__(seed_num)
        self.model = model.to('cpu')
    
    def filter_data(self, *args, **kwargs)->List:
        """ overwriting parent to free model gpu memory after use """
        output = super().filter_data(*args, **kwargs)
        self.model.to('cpu')
        torch.cuda.empty_cache() 
        return output
    
    @staticmethod
    def tensor_prep(ex:dict):
        label = torch.LongTensor([ex['output']])
        inp_id = torch.LongTensor([ex['input'][0]])
        t_id = torch.LongTensor([ex['input'][1]])
        attn_m = torch.LongTensor([ex['input'][2]])
        return inp_id, t_id, attn_m, label
    
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
    def __init__(self, seed_num:int=1, model=None, negate=False):
        super().__init__(seed_num, model)
        self.negate = negate
     
    def get_ex_score(self, ex)->float:
        inp_id, t_id, attn_m, label = self.tensor_prep(ex)
        outputs = self.model(input_ids=inp_id, attention_mask=attn_m, token_type_ids=t_id, labels=label)
        loss = outputs[0].item()
        if self.negate:
            loss = -1.*loss
        return float(loss)
        
class KMeansPruner(ModelDataPruner):
    '''
        View samples in encoder embedding space
        PCA compression of samples
        K-Means clustering
        Select K samples (closest to each cluster mean)
    '''
    def __init__(self, seed_num:int=1, model=None):
        super().__init__(seed_num, model)

     
    def filter_data(self, data:List, ret_frac:float, ncomps:int=10)->List:
        N = int(ret_frac*len(data))
        
        # Encoder embedding space
        H = self.get_hidden_vecs(data)

        # PCA compression
        pca = PCA(n_components=ncomps)
        compressed = pca.fit_transform(H)

        # K means clustering
        kmeans = KMeans(n_clusters=N, random_state=0).fit(compressed)
        
        # Select closest to mean per cluster
        cluster_means = kmeans.cluster_centers_
        selected_inds = []
        for k in N:
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
        return emb.squeeze().detach.cpu()
        


