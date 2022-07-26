'''
Inputs: in-domain training set
        out-domain training set

Function:

        0) train encoder/model on in-domain data
        1) pass in-domain through encoder to get feature space
        2) Plot PCA 2 dim of feature space
        3) save PCA transformation matrix
        4) Plot PCA 2 dim of out domain using saved PCA transformation on out-domain passed through encoder
'''

import torch 
import matplotlib.pyplot as plt
import argparse
import os
import sys
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def get_covariance_matrix(X):
    '''
    Returns the covariance of the data X
    X should contain a single data point per row of the tensor
    '''
    X_mean = torch.mean(X, dim=0)
    X_mean_matrix = torch.outer(X_mean, X_mean)
    X_corr_matrix = torch.matmul(torch.transpose(X, 0, 1), X)/X.size(0)
    cov = X_corr_matrix - X_mean_matrix
    return cov

def get_e_v(cov):
    '''
    Returns eigenvalues and eigenvectors in descending order by eigenvalue size
    '''
    e, v = torch.symeig(cov, eigenvectors=True)
    v = torch.transpose(v, 0, 1)
    e_abs = torch.abs(e)

    inds = torch.argsort(e_abs, descending=True)
    e = e_abs[inds]
    v = v[inds]

    return e,v

def get_pca_comps(feats, num_comps=2, eigenvectors=None, correction_mean=None):
    if eigenvectors is None:
        pass
        # calculate PCA eigenvectors and correction mean
        cov_mat = get_covariance_matrix(feats)
        eigenvectors, _ = get_e_v(cov_mat)
        correction_mean = torch.mean(feats, dim=0)
    with torch.no_grad():
        # Correct by pre-calculated authentic data mean
        X = X - correction_mean.repeat(X.size(0), 1)

        v_map = eigenvectors[:num_comps]
        comps = torch.einsum('bi,ji->bj', X, v_map)
    return comps, eigenvectors, correction_mean

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

def get_hidden_vecs_batched(data, model, device, batch_size=32):
    labels = [d['output'] for d in data]
    input_ids = [d['inputs'][0][l_ind] for d,l_ind in zip(data, labels)]
    token_type_ids = [d['inputs'][1][l_ind] for d,l_ind in zip(data, labels)]
    attention_masks = [d['inputs'][2][l_ind] for d,l_ind in zip(data, labels)]
    
    dl = dl_prep(labels, input_ids, token_type_ids, attention_masks, bs=batch_size) 

    model.to(device)
    model.eval()
    all_embs = []
    for i, (inp_id, tok_typ_id, att_msk, _) in enumerate(dl):
        # print(f'On {i}/{len(dl)}')
        inp_id, tok_typ_id, att_msk = inp_id.to(device), tok_typ_id.to(device), att_msk.to(device)
        with torch.no_grad():
            embs = model.electra(input_ids=inp_id, attention_mask=att_msk, token_type_ids=tok_typ_id)[0]
            embs_cls = embs[:,0,:]
        all_embs.append(embs_cls.detach().cpu())
    H = torch.cat(all_embs)
    return H
    
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get all command line arguments.')
    parser.add_argument('--model_path', type=int, required=True, help='model path')
    parser.add_argument('--train_data_path', type=str, required=True, help='Load path of training data')

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    # Set device
    device = get_default_device()
    
    # Load model

    # Load in domain data

    # Get in-domain features
    feats_in = get_hidden_vecs_batched(data, model, device)

    # Get in-domain pca_comps
    in_comps, eigenvectors, correction_mean = get_pca_comps(feats_in)



    # Load out of domain data

    # Get out-domain features

    # Get out-domain pca_comps
    out_comps, _, _ = get_pca_comps(feats_out, eigenvectors=eigenvectors, correction_mean=correction_mean)


