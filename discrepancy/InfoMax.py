from torch import nn
from utils.utils import to_cuda
import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning.losses import GenericPairLoss
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import common_functions as c_f

from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.reducers import AvgNonZeroReducer
import numpy as np

from pytorch_metric_learning import losses, miners, distances, reducers, testers


class NWJ_ContrastiveLoss(GenericPairLoss):
    def __init__(self, pos_margin=0, neg_margin=1, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.add_to_recordable_attributes(
            list_of_names=["pos_margin", "neg_margin"], is_stat=False
        )

    def _compute_loss(self, pos_pair_dist, neg_pair_dist, indices_tuple):
        pos_loss, neg_loss = 0, 0
        if len(pos_pair_dist) > 0:
            pos_loss = self.get_per_pair_loss(pos_pair_dist, "pos")
        if len(neg_pair_dist) > 0:
            neg_loss = self.get_per_pair_loss(neg_pair_dist, "neg")
        # print(1)
        pos_pairs = lmu.pos_pairs_from_tuple(indices_tuple)
        neg_pairs = lmu.neg_pairs_from_tuple(indices_tuple)
        # 1
        return {
            "pos_loss": {
                "losses": pos_loss,
                "indices": pos_pairs,
                "reduction_type": "pos_pair",
            },
            "neg_loss": {
                "losses": neg_loss,
                "indices": neg_pairs,
                "reduction_type": "neg_pair",
            },
        }

    def get_per_pair_loss(self, pair_dists, pos_or_neg):
        loss_calc_func = self.pos_calc if pos_or_neg == "pos" else self.neg_calc
        margin = self.pos_margin if pos_or_neg == "pos" else self.neg_margin
        per_pair_loss = loss_calc_func(pair_dists, margin)
        return per_pair_loss

    def pos_calc(self, pos_pair_dist, margin):
        return torch.nn.functional.relu(self.distance.margin(pos_pair_dist, margin))

    def neg_calc(self, neg_pair_dist, margin):
        return torch.exp(torch.nn.functional.relu(self.distance.margin(margin, neg_pair_dist)))/np.exp(1)-1/np.exp(1)

    def get_default_reducer(self):
        return reducers.DoNothingReducer()

    def _sub_loss_names(self):
        return ["pos_loss", "neg_loss"]



class InfoMax(object):
    def __init__(self, num_layers,
                 num_classes, **kwargs):


        self.num_classes = num_classes
        self.num_layers = num_layers

    def Infomax_loss(self, source, target, nums_S, nums_T,label_source,label_target,info_coef,
                     margin,pos_margin,neg_margin,type_of_triplets,
                     cdd_on_logidts,info_on_logidts,square_on_logidts,info_coef_2,info_coef_square,margin_2,pos_margin_2,neg_margin_2,distance_layers,miner,INFOMETHOD):

        assert (len(nums_S) == len(nums_T)), \
            "The number of classes for source (%d) and target (%d) should be the same." \
            % (len(nums_S), len(nums_T))

        num_classes = len(nums_S)

        label1 = []
        label2=[]
        for i in range(num_classes):
            for j in range(nums_S[i]):
                label1.append(i)
        for i in range(num_classes):
            for j in range(nums_T[i]):
                label2.append(i)
        label1 = to_cuda(torch.tensor(label1))
        label2=to_cuda(torch.tensor(label2))
        label=torch.cat([label1,label2],0)

        data = torch.cat([source[0], target[0]], 0)

        report={}

        if INFOMETHOD=='NCE':
            loss_func=losses.NTXentLoss().cuda()
            loss_info = loss_func(data, label)

        elif INFOMETHOD=='NWJ':
            distance=getattr(distances,distance_layers[0])()
            loss_func=NWJ_ContrastiveLoss(pos_margin=pos_margin, neg_margin=neg_margin,distance=distance,reducer=reducers.DoNothingReducer()).cuda()
            if miner=='PairMarginMiner':
                mining_func = miners.PairMarginMiner(pos_margin=pos_margin, neg_margin=neg_margin, distance=distance)
            elif miner=='TripletMarginMiner':
                mining_func = miners.TripletMarginMiner(margin=margin, distance=distance, type_of_triplets=type_of_triplets)
            indices_tuple = mining_func(data, label)
            # loss = loss_func(embeddings, labels,indices_tuple)['loss']
            loss_dic = loss_func(data, label, indices_tuple)
            try:
                loss_pos = loss_dic['pos_loss']['losses'].sum()/((loss_dic['pos_loss']['losses']>0).sum()+1e-8)
                loss_neg = loss_dic['neg_loss']['losses'].sum()/((loss_dic['neg_loss']['losses']>0).sum()+1e-8)
            except:
                loss_pos = loss_dic['pos_loss']['losses']
                loss_neg = loss_dic['neg_loss']['losses']
            loss_info = loss_pos + loss_neg
            report['loss_pos']=float(loss_pos)
            report['loss_neg'] = float(loss_neg)
        loss=info_coef *loss_info


        if info_on_logidts:
            data = torch.cat([source[1], target[1]], 0)
            if INFOMETHOD == 'NCE':
                loss_func = losses.NTXentLoss().cuda()
                loss_2 = loss_func(data, label)
            elif INFOMETHOD == 'NWJ':
                distance=getattr(distances,distance_layers[1])()
                loss_func = NWJ_ContrastiveLoss(pos_margin=pos_margin_2, neg_margin=neg_margin_2,distance=distance,reducer=reducers.DoNothingReducer()).cuda()
                if miner == 'TripletMarginMiner':
                    mining_func = miners.TripletMarginMiner(margin=margin_2, distance=distance,
                                                            type_of_triplets=type_of_triplets)
                indices_tuple = mining_func(data, label)
                # loss = loss_func(embeddings, labels,indices_tuple)['loss']
                loss_dic = loss_func(data, label, indices_tuple)
                try:
                    loss_pos = loss_dic['pos_loss']['losses'].sum() / ((loss_dic['pos_loss']['losses'] > 0).sum() + 1e-8)
                    loss_neg = loss_dic['neg_loss']['losses'].sum() / ((loss_dic['neg_loss']['losses'] > 0).sum() + 1e-8)
                except:
                    loss_pos = loss_dic['pos_loss']['losses']
                    loss_neg = loss_dic['neg_loss']['losses']
                loss_2 = loss_pos + loss_neg
                report['loss_pos_2'] = float(loss_pos)
                report['loss_neg_2'] = float(loss_neg)
            loss = loss+info_coef_2*loss_2

        if square_on_logidts:
            label_target=torch.cat(label_target,0).cuda()
            label_onehot = torch.scatter(torch.zeros([label_target.shape[0],
                                                       target[1].shape[1]]).cuda(), 1,
                                          label_target.unsqueeze(1),
                                          torch.ones([label_target.shape[0], 1]).cuda())
            # loss = loss_func(torch.cat([probs,source_onehot],0), torch.cat([labels,labels],0))
            loss_2 = (target[1] - label_onehot).square().sum(1).mean(0)
            loss = loss + info_coef_square * loss_2



        return loss,loss_info,loss_2,report
