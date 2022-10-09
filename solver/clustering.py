import torch
from torch.nn import functional as F
from utils.utils import to_cuda, to_onehot
from scipy.optimize import linear_sum_assignment
from math import ceil
from aux_function import JS_divergence,correct_rate_func
import numpy as np
import time
from pytorch_metric_learning import distances

class DIST(object):
    def __init__(self, dist_type='cos'):
        self.dist_type = dist_type 

    def get_dist(self, pointA, pointB, cross=False):
        return getattr(self, self.dist_type)(
		pointA, pointB, cross)

    def cos(self, pointA, pointB, cross):
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)
        if not cross:
            return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
        else:
            NA = pointA.size(0)
            NB = pointB.size(0)
            assert(pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))

class Clustering(object):
    def __init__(self, eps, feat_key, max_len=1000, dist_type='cos'):
        self.eps = eps
        self.Dist = DIST(dist_type)
        self.samples = {}
        self.path2label = {}
        self.center_change = None
        self.stop = False
        self.feat_key = feat_key
        self.max_len = max_len

    def set_init_centers(self, init_centers):
        self.centers = init_centers
        self.init_centers = init_centers
        self.num_classes = self.centers.size(0)

    def clustering_stop(self, centers):
        if centers is None:
            self.stop = False
        else:
            dist = self.Dist.get_dist(centers, self.centers) 
            dist = torch.mean(dist, dim=0)
            print('dist %.4f' % dist.item())
            self.stop = dist.item() < self.eps

    def assign_labels(self, feats):
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=1)
        return dists, labels

    def align_centers(self):
        cost = self.Dist.get_dist(self.centers, self.init_centers, cross=True)
        cost = cost.data.cpu().numpy()
        _, col_ind = linear_sum_assignment(cost)
        return col_ind

    def collect_samples(self, net, loader):
        data_feat, data_gt, data_paths = [], [], []
        for sample in iter(loader): 
            data = sample['Img'].cuda()
            data_paths += sample['Path']
            if 'Label' in sample.keys():
                data_gt += [to_cuda(sample['Label'])]

            output = net.forward(data)
            feature = output[self.feat_key].data 
            data_feat += [feature]
            
        self.samples['data'] = data_paths
        self.samples['gt'] = torch.cat(data_gt, dim=0) \
                    if len(data_gt)>0 else None
        self.samples['feature'] = torch.cat(data_feat, dim=0)

    def feature_clustering(self, net, loader):
        centers = None 
        self.stop = False 

        self.collect_samples(net, loader)
        feature = self.samples['feature']

        refs = to_cuda(torch.LongTensor(range(self.num_classes)).unsqueeze(1))
        num_samples = feature.size(0)
        num_split = ceil(1.0 * num_samples / self.max_len)

        while True:
            self.clustering_stop(centers)
            if centers is not None:
                self.centers = centers
            if self.stop: break

            centers = 0
            count = 0

            start = 0 
            for N in range(num_split):
                cur_len = min(self.max_len, num_samples - start)
                cur_feature = feature.narrow(0, start, cur_len)
                dist2center, labels = self.assign_labels(cur_feature)
                labels_onehot = to_onehot(labels, self.num_classes)
                count += torch.sum(labels_onehot, dim=0)
                labels = labels.unsqueeze(0)
                mask = (labels == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
                reshaped_feature = cur_feature.unsqueeze(0)    
                # update centers
                centers += torch.sum(reshaped_feature * mask, dim=1)
                start += cur_len
    
            mask = (count.unsqueeze(1) > 0).type(torch.cuda.FloatTensor) 
            centers = mask * centers + (1 - mask) * self.init_centers
            
        dist2center, labels = [], []
        start = 0
        count = 0
        for N in range(num_split):
            cur_len = min(self.max_len, num_samples - start)
            cur_feature = feature.narrow(0, start, cur_len)
            cur_dist2center, cur_labels = self.assign_labels(cur_feature)

            labels_onehot = to_onehot(cur_labels, self.num_classes)
            count += torch.sum(labels_onehot, dim=0)

            dist2center += [cur_dist2center]
            labels += [cur_labels]
            start += cur_len

        self.samples['label'] = torch.cat(labels, dim=0)
        self.samples['dist2center'] = torch.cat(dist2center, dim=0)

        cluster2label = self.align_centers()
        # reorder the centers
        self.centers = self.centers[cluster2label, :]
        # re-label the data according to the index
        num_samples = len(self.samples['feature'])
        for k in range(num_samples):
            self.samples['label'][k] = cluster2label[self.samples['label'][k]].item()

        self.center_change = torch.mean(self.Dist.get_dist(self.centers, \
                    self.init_centers))

        for i in range(num_samples):
            self.path2label[self.samples['data'][i]] = self.samples['label'][i].item()

        del self.samples['feature']





    def feature_clustering_weighted(self, net, loader, weight_by_label, top_k, prop_max_step, prop_alpha, graph_method,
                                    thre_filter,JS_report=False):
        centers = None
        self.stop = False
        # tem=time.time()
        self.collect_samples(net, loader)

        feature = self.samples['feature']

        refs = to_cuda(torch.LongTensor(range(self.num_classes)).unsqueeze(1))
        num_samples = feature.size(0)
        num_split = ceil(1.0 * num_samples / self.max_len)

        while True:
            self.clustering_stop(centers)
            if centers is not None:
                self.centers = centers
            if self.stop: break

            centers = 0
            count = 0

            start = 0
            for N in range(num_split):
                cur_len = min(self.max_len, num_samples - start)
                cur_feature = feature.narrow(0, start, cur_len)
                dist2center, labels = self.assign_labels(cur_feature)
                labels_onehot = to_onehot(labels, self.num_classes)
                count += torch.sum(labels_onehot, dim=0)
                labels = labels.unsqueeze(0)
                mask = (labels == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
                reshaped_feature = cur_feature.unsqueeze(0)
                # update centers
                centers += torch.sum(reshaped_feature * mask, dim=1)
                start += cur_len

            mask = (count.unsqueeze(1) > 0).type(torch.cuda.FloatTensor)
            centers = mask * centers + (1 - mask) * self.init_centers

        dist2center, labels = [], []
        start = 0
        count = 0
        for N in range(num_split):
            cur_len = min(self.max_len, num_samples - start)
            cur_feature = feature.narrow(0, start, cur_len)
            cur_dist2center, cur_labels = self.assign_labels(cur_feature)

            labels_onehot = to_onehot(cur_labels, self.num_classes)
            count += torch.sum(labels_onehot, dim=0)

            dist2center += [cur_dist2center]
            labels += [cur_labels]
            start += cur_len

        self.samples['label'] = torch.cat(labels, dim=0)
        self.samples['dist2center'] = torch.cat(dist2center, dim=0)

        cluster2label = self.align_centers()
        # reorder the centers
        self.centers = self.centers[cluster2label, :]
        # re-label the data according to the index
        num_samples = len(self.samples['feature'])
        for k in range(num_samples):
            self.samples['label'][k] = cluster2label[self.samples['label'][k]].item()

        self.center_change = torch.mean(self.Dist.get_dist(self.centers, \
                                                           self.init_centers))

        for i in range(num_samples):
            self.path2label[self.samples['data'][i]] = self.samples['label'][i].item()


        tem=time.time()


        if weight_by_label:
            self.samples['weight'] = torch.scatter(torch.zeros([len(self.samples['data']),
                                                                self.num_classes]).cuda(), 1,
                                                   self.samples['label'].unsqueeze(1),
                                                   torch.ones([len(self.samples['data']), 1]).cuda())
        else:
            self.samples['weight'] = feature.shape[0] * torch.exp(-self.samples['dist2center'].pow(2)) / torch.exp(
                -self.samples['dist2center'].pow(2)).sum(0, keepdim=True)


        labels = self.samples['label']
        gt = self.samples['gt']
        dist_min = torch.min(self.samples['dist2center'], dim=1)[0]
        if JS_report:
            JS_rec = []
            sample_filterd = {}
            sample_filterd['feature'] = self.samples['feature']
            sample_filterd['graph'] = get_graph(sample_filterd, top_k, graph_method)
            for thre in np.linspace(0, 0.14, 30):
                index = dist_min < thre
                labels_se = torch.masked_select(labels, index)
                gt_se = torch.masked_select(gt, index)
                cr = (labels_se.cuda() == gt_se.cuda()).float().mean()
                class_num = (to_onehot(labels_se, 31).sum(0) > 0).sum()



                sample_filterd['weight'] = self.samples['weight'] * index.unsqueeze(1)
                rec = label_propagation(sample_filterd, prop_max_step, prop_alpha)
                JS_tem = []
                for i in range(len(rec)):
                    preds = rec[i]
                    gts = self.samples['gt']
                    res = correct_rate_func(preds, gts)
                    tem = torch.scatter(torch.zeros_like(self.samples['weight']).cuda(), 1,
                                        gts.unsqueeze(1).cuda(),
                                        torch.ones([len(self.samples['data']), 1]).cuda())
                    JS = JS_divergence(preds, tem)
                    # print('proped %d %s: %.4f , JS: %.3f ' % (i, 'accuracy', res, JS.mean()))
                    JS_tem.append(JS.mean())
                JS_rec.append(JS_tem)
            JS_rec = np.asarray(JS_rec)
            JS_rec = np.mean(JS_rec, 0)
            print(JS_rec)
        # print('weight finished ',time.time()-tem)
        tem = time.time()
        index = dist_min < thre_filter
        self.samples['graph'],self.samples['dist']=get_graph(self.samples, top_k,self.max_len, graph_method)
        self.samples['weight'] = self.samples['weight'] * index.unsqueeze(1)

        return



def get_graph(samples, top_k, max_len, graph_method='KNN'):

    distance = distances.LpDistance()
    n_sample = samples['feature'].shape[0]
    feature = samples['feature'].detach().cpu()
    dist=distance(feature,feature)

    if graph_method == 'KNN':
        inds = torch.topk(dist, top_k, dim=1, largest=False).indices
        A = torch.scatter(torch.zeros(n_sample, n_sample), 1, inds, torch.ones(n_sample, top_k))
        A = torch.scatter(A, 0, inds.T, torch.ones(top_k, n_sample))
        D_inv = A.sum(1,keepdim=True).pow(-0.5)
        W = D_inv * A * D_inv.T
    return W,dist

def label_propagation(samples, prop_max_step=5, alpha=0.9):
    W=samples['graph'].cpu()
    rec = []
    Y = samples['weight'].cpu()
    F = Y
    rec.append(F)
    for i in range(prop_max_step):
        F = alpha * W @ F + (1 - alpha) * Y
        rec.append(F)
    return rec
