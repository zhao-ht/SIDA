from __future__ import print_function


import numpy as np

import torch
import torch.nn.parallel

import torch.utils.data

def correct_rate_func(out,label):
    return (torch.argmax(out,1)==label).float().mean()


def Entropy(input_,dim=1):
    bs = input_.size(0)
    input_=input_/input_.sum(dim=dim)
    epsilon = 1e-8
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=dim)
    return entropy




def JS_divergence(P,Q):
    P=P/(P.sum(0,keepdim=True)+1e-10)
    Q=Q/(Q.sum(0,keepdim=True)+1e-10)
    tem=torch.log((P+Q)/2+1e-10)
    JS=(P*((P+1e-10).log()-tem)).sum(0)+(Q*((Q+1e-10).log()-tem)).sum(0)
    return JS.detach().cpu().numpy()


def project_onto_unit_simplex(prob):
    """
    Project an n-dim vector prob to the simplex Dn s.t.
    Dn = { x : x n-dim, 1 >= x >= 0, sum(x) = 1}
    :param prob: a numpy array. Each element is a probability.
    :return: projected probability
    """
    prob_length = len(prob)
    bget = False
    sorted_prob = -np.sort(-prob)
    tmpsum = 0

    for i in range(1, prob_length):
        tmpsum = tmpsum + sorted_prob[i-1]
        tmax = (tmpsum - 1) / i
        if tmax >= sorted_prob[i]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + sorted_prob[prob_length-1] - 1) / prob_length

    return np.maximum(0, prob - tmax)

def project_onto_unit_simplex_matrix(F):
    if isinstance(F,torch.Tensor):
        F_np=F.detach().cpu().numpy()
    else:
        F_np=F
    rec=[]
    for i in range(F.shape[1]):
        rec.append(project_onto_unit_simplex(F_np[:,i]).reshape([-1,1]))
    res=np.concatenate(rec,axis=1)
    return res