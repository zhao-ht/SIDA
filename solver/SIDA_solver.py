import torch
import torch.nn as nn
import os
from . import utils as solver_utils
from utils.utils import to_cuda, to_onehot
from torch import optim
from . import clustering
from discrepancy.InfoMax import InfoMax
from math import ceil as ceil
from .base_solver import BaseSolver
from copy import deepcopy
import numpy as np


from solver.clustering import DIST, label_propagation, get_graph
from aux_function import project_onto_unit_simplex, project_onto_unit_simplex_matrix, Entropy, JS_divergence
from pytorch_metric_learning import distances
import time


class SIDASolver(BaseSolver):
    def __init__(self, net, dataloader, bn_domain_map={}, resume=None, **kwargs):
        super(SIDASolver, self).__init__(net, dataloader, \
                                        bn_domain_map=bn_domain_map, resume=resume, **kwargs)

        if len(self.bn_domain_map) == 0:
            self.bn_domain_map = {self.source_name: 0, self.target_name: 1}

        self.clustering_source_name = 'clustering_' + self.source_name
        self.clustering_target_name = 'clustering_' + self.target_name

        assert ('categorical' in self.train_data)

        num_layers = len(self.net.module.FC) + 1
        self.infomatric = InfoMax(
                       num_layers=num_layers, num_classes=self.opt.DATASET.NUM_CLASSES,
                       )


        self.clustering = clustering.Clustering(self.opt.CLUSTERING.EPS,
                                                self.opt.CLUSTERING.FEAT_KEY,
                                                self.opt.CLUSTERING.BUDGET)

        self.clustered_target_samples = {}


        if self.opt.SIDA.CONSISTENCY == 'infomax':
            self.consistency = self.infomatric.Infomax_loss
        self.best_acc = 0

    def complete_training(self):
        if self.loop >= self.opt.TRAIN.MAX_LOOP:
            return True

        if 'target_centers' not in self.history or \
                'ts_center_dist' not in self.history or \
                'target_labels' not in self.history:
            return False

        if len(self.history['target_centers']) < 2 or \
                len(self.history['ts_center_dist']) < 1 or \
                len(self.history['target_labels']) < 2:
            return False

        # target centers along training
        target_centers = self.history['target_centers']
        eval1 = torch.mean(self.clustering.Dist.get_dist(target_centers[-1],
                                                         target_centers[-2])).item()

        # target-source center distances along training
        eval2 = self.history['ts_center_dist'][-1].item()

        # target labels along training
        path2label_hist = self.history['target_labels']
        paths = self.clustered_target_samples['data']
        num = 0
        for path in paths:
            pre_label = path2label_hist[-2][path]
            cur_label = path2label_hist[-1][path]
            if pre_label != cur_label:
                num += 1
        eval3 = 1.0 * num / len(paths)

        return (eval1 < self.opt.TRAIN.STOP_THRESHOLDS[0] and \
                eval2 < self.opt.TRAIN.STOP_THRESHOLDS[1] and \
                eval3 < self.opt.TRAIN.STOP_THRESHOLDS[2])

    def compute_iters_per_loop_weight_classaware(self, filtered_classes):
        self.iters_per_loop = int(
            len(self.train_data['categorical_weight']['loader'])) * self.opt.TRAIN.UPDATE_EPOCH_PERCENTAGE
        # print('Iterations in one loop: %d' % (self.iters_per_loop))
        return self.iters_per_loop

    def filter_samples_weight_classaware(self, samples, threshold=0.05):
        batch_size_full = len(samples['data'])
        min_dist = torch.min(samples['dist2center'], dim=1)[0]

        mask = min_dist < threshold
        index = torch.where(min_dist < threshold)[0]

        filtered_data = [samples['data'][m]
                         for m in range(mask.size(0)) if mask[m].item() == 1]
        filtered_label = torch.masked_select(samples['label'], mask)
        filtered_gt = torch.masked_select(samples['gt'], mask) \
            if samples['gt'] is not None else None
        filtered_weight = torch.index_select(samples['weight'], 0, index)

        filtered_samples = {}
        filtered_samples['data'] = filtered_data
        filtered_samples['label'] = filtered_label
        filtered_samples['gt'] = filtered_gt
        filtered_samples['weight'] = filtered_weight

        assert len(filtered_samples['data']) == filtered_samples['label'].size(0)
        print('select %f' % (1.0 * len(filtered_data) / batch_size_full))

        return filtered_samples

    def filtering_weight_classaware(self):
        threshold = self.opt.CLUSTERING.FILTERING_THRESHOLD
        min_sn_cls = self.opt.TRAIN.MIN_SN_PER_CLASS
        target_samples = self.clustered_target_samples

        # filtering the samples
        chosen_samples = self.filter_samples_weight_classaware(
            target_samples, threshold=threshold)

        # filtering the classes
        filtered_classes = solver_utils.filter_class(
            chosen_samples['label'], min_sn_cls, self.opt.DATASET.NUM_CLASSES)

        print('The number of filtered classes: %d.' % len(filtered_classes))
        return chosen_samples, filtered_classes

    def filtering_importantweight_classaware(self):

        class_entropy = Entropy(self.clustered_target_samples['weight'], 0)
        chosen_class = list(torch.where(class_entropy >= self.opt.SIDA.MIN_ENTROPY_PER_CLASS)[0].cpu().numpy())
        print('filtered class: ', len(chosen_class))
        return chosen_class

    def split_samples_classwise_weight_classaware(self, samples, num_classes):
        data = samples['data']
        label = samples['label']
        gt = samples['gt']
        weight = samples['weight']
        samples_list = []
        for c in range(num_classes):
            mask = (label == c)
            index = torch.where(label == c)[0]
            data_c = [data[k] for k in range(mask.size(0)) if mask[k].item() == 1]
            label_c = torch.masked_select(label, mask)
            gt_c = torch.masked_select(gt, mask) if gt is not None else None
            weight_c = torch.index_select(weight, 0, index) if gt is not None else None
            samples_c = {}
            samples_c['data'] = data_c
            samples_c['label'] = label_c
            samples_c['gt'] = gt_c
            samples_c['weight'] = weight_c.detach().cpu()
            samples_list.append(samples_c)

        return samples_list

    def construct_categorical_dataloader_weight_classaware(self, samples, filtered_classes):
        # update self.dataloader
        target_classwise = self.split_samples_classwise_weight_classaware(
            samples, self.opt.DATASET.NUM_CLASSES)

        dataloader = self.train_data['categorical_weight']['loader']
        classnames = dataloader.classnames
        dataloader.class_set = [classnames[c] for c in filtered_classes]
        dataloader.target_paths = {classnames[c]: target_classwise[c]['data'] \
                                   for c in filtered_classes}
        dataloader.num_selected_classes = min(self.opt.TRAIN.NUM_SELECTED_CLASSES, len(filtered_classes))
        dataloader.weight = {classnames[c]: target_classwise[c]['weight'] \
                             for c in filtered_classes}
        dataloader.construct()

    def construct_categorical_dataloader_weight_classaware_importantweight(self, samples, filtered_classes):
        dataloader = self.train_data['categorical_weight']['loader']
        classnames = dataloader.classnames
        dataloader.class_set = [classnames[c] for c in filtered_classes]
        dataloader.target_paths = samples['data']
        dataloader.num_selected_classes = min(self.opt.TRAIN.NUM_SELECTED_CLASSES, len(filtered_classes))
        dataloader.weight = torch.index_select(samples['weight'], 1, torch.tensor(filtered_classes).cuda()).cpu()
        dataloader.construct()

    def update_network_weight_classaware(self, filtered_classes):
        # initial configuration
        stop = False
        update_iters = 0

        self.train_data[self.source_name]['iterator'] = \
            iter(self.train_data[self.source_name]['loader'])
        self.train_data['categorical_weight']['iterator'] = \
            iter(self.train_data['categorical_weight']['loader'])

        while not stop:
            # update learning rate
            self.update_lr()

            # set the status of network
            self.net.train()
            self.net.zero_grad()

            loss = 0
            ce_loss_iter = 0
            cdd_loss_iter = 0
            loss_1_iter = 0
            loss_2_iter = 0

            # coventional sampling for training on labeled source data
            source_sample = self.get_samples(self.source_name)
            source_data, source_gt = source_sample['Img'], \
                                     source_sample['Label']

            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)
            self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
            source_preds = self.net(source_data)['logits']

            # compute the cross-entropy loss
            ce_loss = self.opt.SIDA.CLASSIFYCOEF * self.CELoss(source_preds, source_gt)
            ce_loss.backward()
            ce_loss_iter += float(ce_loss)
            loss += self.opt.SIDA.CLASSIFYCOEF * float(ce_loss)

            if len(filtered_classes) > 0 and not self.opt.MODEL.TRAIN_SOURCE_ONLY:
                # update the network parameters
                # 1) class-aware sampling
                data_loader = self.train_data['categorical_weight']['loader']
                data_iterator = self.train_data['categorical_weight']['iterator']
                assert data_loader is not None and data_iterator is not None, \
                    'Check your dataloader of %s.' % 'categorical_weight'

                try:
                    samples = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(data_loader)
                    samples = next(data_iterator)
                    self.train_data['categorical_weight']['iterator'] = data_iterator

                # samples['Img_source'],samples['Label_source'],samples['weight_source'],\
                # samples['Img_target'],samples['Label_target'],samples['weight_target']
                nums_S = [len(tem) for tem in samples['Img_source']]
                nums_T = [len(tem) for tem in samples['Img_target']]
                # 2) forward and compute the loss
                source_cls_concat = torch.cat([to_cuda(samples)
                                               for samples in samples['Img_source']], dim=0)
                target_cls_concat = torch.cat([to_cuda(samples)
                                               for samples in samples['Img_target']], dim=0)

                self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
                feats_source = self.net(source_cls_concat)
                self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                feats_target = self.net(target_cls_concat)

                # prepare the features
                feats_toalign_S = [feats_source[key] for key in feats_source if
                                   key in self.opt.SIDA.ALIGNMENT_FEAT_KEYS]
                feats_toalign_T = [feats_target[key] for key in feats_target if
                                   key in self.opt.SIDA.ALIGNMENT_FEAT_KEYS]


                if self.opt.SIDA.CONSISTENCY == 'infomax':
                    cdd_loss, loss_1, loss_2, report = self.infomatric.Infomax_loss(feats_toalign_S, feats_toalign_T,
                                                                             nums_S, nums_T, samples['Label_source'],
                                                                             samples['Label_source'],
                                                                             self.opt.SIDA.INFOCOEF,
                                                                             self.opt.SIDA.MARGIN,
                                                                             self.opt.SIDA.POS_MARGIN,
                                                                             self.opt.SIDA.NEG_MARGIN,
                                                                             self.opt.SIDA.TYPE_OF_TRIPLETS,
                                                                             self.opt.SIDA.CDDONLOGITS,
                                                                             self.opt.SIDA.INFOONLOGITS,
                                                                             self.opt.SIDA.SQUAREONLOGITS,
                                                                             self.opt.SIDA.INFOCOEF_2,
                                                                             self.opt.SIDA.INFOCOEF_SQUARE,
                                                                             self.opt.SIDA.MARGIN_2,
                                                                             self.opt.SIDA.POS_MARGIN_2,
                                                                             self.opt.SIDA.NEG_MARGIN_2,
                                                                             self.opt.SIDA.DISTANCES,
                                                                             self.opt.SIDA.MINER,
                                                                             self.opt.SIDA.INFOMETHOD
                                                                             )

                cdd_loss *= self.opt.CDD.LOSS_WEIGHT
                cdd_loss.backward()

                cdd_loss_iter += float(cdd_loss)
                loss_1_iter += float(loss_1)
                loss_2_iter += float(loss_2)
                loss += float(cdd_loss)

            # update the network
            self.optimizer.step()

            if self.opt.TRAIN.LOGGING and (update_iters + 1) % \
                    (max(1, self.iters_per_loop // self.opt.TRAIN.NUM_LOGGING_PER_LOOP)) == 0:
                accu = self.model_eval(source_preds, source_gt)
                cur_loss = {'ce_loss': ce_loss_iter, 'cdd_loss': cdd_loss_iter,
                            'loss_1': loss_1_iter, 'loss_2': loss_2_iter,
                            'total_loss': loss}
                self.logging(cur_loss, accu)
                # if self.opt.SIDA.CONSISTENCY == 'infomax' and self.opt.SIDA.INFOMETHOD == 'NWJ' and not self.opt.MODEL.TRAIN_SOURCE_ONLY:
                #     print(report)

            self.opt.TRAIN.TEST_INTERVAL = min(1.0, self.opt.TRAIN.TEST_INTERVAL)
            self.opt.TRAIN.SAVE_CKPT_INTERVAL = min(1.0, self.opt.TRAIN.SAVE_CKPT_INTERVAL)

            if self.opt.TRAIN.TEST_INTERVAL > 0 and \
                    (update_iters + 1) % int(self.opt.TRAIN.TEST_INTERVAL * self.iters_per_loop) == 0:
                with torch.no_grad():
                    self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                    accu = self.test()
                    self.best_acc = max(accu, self.best_acc)
                    print('Test at (loop %d, iters: %d) with %s: %.4f. best: %.4f.' % (self.loop,
                                                                                       self.iters, self.opt.EVAL_METRIC,
                                                                                       accu, self.best_acc))

            if self.opt.TRAIN.SAVE_CKPT_INTERVAL > 0 and \
                    (update_iters + 1) % int(self.opt.TRAIN.SAVE_CKPT_INTERVAL * self.iters_per_loop) == 0:
                self.save_ckpt()

            update_iters += 1
            self.iters += 1

            # update stop condition
            if update_iters >= self.iters_per_loop:
                stop = True
            else:
                stop = False

    def update_labels_Weighted(self):

        tem = time.time()
        net = self.net
        net.eval()
        opt = self.opt

        source_dataloader = self.train_data[self.clustering_source_name]['loader']
        net.module.set_bn_domain(self.bn_domain_map[self.source_name])

        source_centers, self.source_samples = solver_utils.get_centers(net,
                                                                       source_dataloader, self.opt.DATASET.NUM_CLASSES,
                                                                       self.opt.CLUSTERING.FEAT_KEY)

        init_target_centers = source_centers

        # print('source samples finished',time.time()-tem)

        target_dataloader = self.train_data[self.clustering_target_name]['loader']
        net.module.set_bn_domain(self.bn_domain_map[self.target_name])

        self.clustering.set_init_centers(init_target_centers)
        self.clustering.feature_clustering_weighted(net, target_dataloader, self.opt.SIDA.WEIGHT_BY_LABEL,
                                                    self.opt.SIDA.TOP_K,
                                                    self.opt.SIDA.PROP_MAX_STEP,
                                                    self.opt.SIDA.PROP_ALPHA,
                                                    self.opt.SIDA.GRAPH_METHOD,
                                                    self.opt.CLUSTERING.FILTERING_THRESHOLD)
        return self.clustering.samples

    def q_optimization_adv(self):
        cdd_rec = []
        for i in range(len(self.clustered_target_samples['weight_proped'])):
            self.clustered_target_samples['weight'] = self.clustered_target_samples['weight_proped'][i]
            # update dataloaders
            if self.opt.FIXED.IMPORTANTWEIGHT:
                filtered_classes = self.filtering_importantweight_classaware()
                self.construct_categorical_dataloader_weight_classaware_importantweight(
                    self.clustered_target_samples, filtered_classes)
            else:
                target_hypt, filtered_classes = self.filtering_weight_classaware()
                self.construct_categorical_dataloader_weight_classaware(target_hypt, filtered_classes)
            # update train data setting
            self.iters_per_loop = self.compute_iters_per_loop_weight_classaware(filtered_classes)
            cdd = float(self.compute_consistancy(filtered_classes))
            # print('proped {} ,cdd {}'.format(i, cdd))
            cdd_rec.append(cdd)
        ind = np.argmax(cdd_rec)
        self.clustered_target_samples['weight'] = self.clustered_target_samples['weight_proped'][ind]

    def q_optimization_prop(self):
        samples = {}
        N_source = 10000
        if len(self.source_samples['feature']) > N_source:
            indices_source = torch.randperm(len(self.source_samples['feature']))[:N_source]
            source_samples = self.source_samples['feature'][indices_source]
            source_label = self.source_samples['gt'][indices_source]
        else:
            N_source = len(self.source_samples['feature'])
            source_samples = self.source_samples['feature']
            source_label = self.source_samples['gt']

        samples['feature'] = torch.cat([source_samples,
                                        self.clustered_target_samples['feature']], 0).cpu()
        adjancent_matrix = self.clustered_target_samples['graph'].cpu()

        Dist = distances.LpDistance()
        score = 1 - Dist(samples['feature'], samples['feature'])


        rec = []
        initial_distribution = self.clustered_target_samples['weight'].cpu()
        NY = self.opt.DATASET.NUM_CLASSES
        source_onehot = torch.scatter(torch.zeros([len(source_label),
                                                   NY]), 1,
                                      source_label.cpu().unsqueeze(1),
                                      torch.ones([len(source_label), 1]))
        F = initial_distribution
        rec.append(F)

        ones = torch.ones([NY, NY])
        alpha = self.opt.SIDA.PROP_ALPHA
        beta = self.opt.SIDA.PROP_BETA
        if self.opt.SIDA.PROP_ADVER:
            beta = -beta

        F = F / (F.sum(0, keepdim=True) + 1e-8)
        product_rec = []
        for i in range(self.opt.SIDA.PROP_MAX_STEP):
            PQ = torch.cat([source_onehot / torch.sum(source_onehot, 0, keepdim=True),
                            F / (torch.sum(F, 0, keepdim=True) + 1e-8)], 0) / 2
            grad_info = score @ PQ / NY \
                        - 1 / np.exp(1) * torch.exp(score) @ PQ @ ones / NY / NY
            grad_info = grad_info[N_source:]
            grad_info = grad_info - grad_info.mean(0, keepdim=True)
            grad_harmo = F - adjancent_matrix @ F
            product_rec.append(100 * float((grad_info * grad_harmo).sum()))
            F = F - alpha * grad_harmo \
                + beta * (grad_harmo.abs() * grad_info)
            F = torch.tensor(project_onto_unit_simplex_matrix(F)).cpu()
            rec.append(F)

        gts = self.clustered_target_samples['gt']
        tem = torch.scatter(torch.zeros_like(F).cuda(), 1,
                            gts.unsqueeze(1).cuda(),
                            torch.ones([len(F), 1]).cuda())
        for i in range(len(product_rec)):
            preds = rec[i]
            JS = JS_divergence(preds.cuda(), tem)
            # print('proped %d  JS: %.3f  Product(*100): %.3f' % (i, JS.mean(), product_rec[i]))
        preds = rec[-1]
        JS = JS_divergence(preds.cuda(), tem)
        # print('proped %d  JS: %.3f' % (len(rec) - 1, JS.mean()))

        self.clustered_target_samples['weight'] = rec[-1].cuda()

    def solve_Weighted(self):
        stop = False
        if self.resume:
            self.iters += 1
            self.loop += 1

        while True:
            # updating the target label hypothesis through clustering

            with torch.no_grad():

                # self.update_ss_alignment_loss_weight()
                print('Clustering based on %s...' % self.source_name)
                self.clustered_target_samples = self.update_labels_Weighted()
                target_centers = self.clustering.centers
                center_change = self.clustering.center_change
                path2label = self.clustering.path2label

                # updating the history
                self.register_history('target_centers', target_centers,
                                      self.opt.CLUSTERING.HISTORY_LEN)
                self.register_history('ts_center_dist', center_change,
                                      self.opt.CLUSTERING.HISTORY_LEN)
                self.register_history('target_labels', path2label,
                                      self.opt.CLUSTERING.HISTORY_LEN)

                if self.clustered_target_samples is not None and \
                        self.clustered_target_samples['gt'] is not None:
                    preds = to_onehot(self.clustered_target_samples['label'],
                                      self.opt.DATASET.NUM_CLASSES)
                    gts = self.clustered_target_samples['gt']
                    res = self.model_eval(preds, gts)
                    # print('Clustering %s: %.4f' % (self.opt.EVAL_METRIC, res))

                # check if meet the stop condition
                stop = self.complete_training()
                if stop: break

            if self.opt.CLUSTERING.SAVE:
                torch.save(self.clustered_target_samples, os.path.join(self.opt.SAVE_DIR,'noisy_pseudo_label'+str(self.loop)))
                torch.save(self.source_samples, os.path.join(self.opt.SAVE_DIR,'source'+str(self.loop)))

            if self.opt.SIDA.Q_OPTIMIZATION_METHOD == 'SEARCH':
                self.q_optimization_adv()
            elif self.opt.SIDA.Q_OPTIMIZATION_METHOD == 'PROPAGATION':
                self.q_optimization_prop()
            else:
                pass

            # update dataloaders
            if self.opt.FIXED.IMPORTANTWEIGHT:
                filtered_classes = self.filtering_importantweight_classaware()
                self.construct_categorical_dataloader_weight_classaware_importantweight(
                    self.clustered_target_samples, filtered_classes)
            else:
                target_hypt, filtered_classes = self.filtering_weight_classaware()
                self.construct_categorical_dataloader_weight_classaware(target_hypt, filtered_classes)

            # update train data setting
            self.iters_per_loop = self.compute_iters_per_loop_weight_classaware(filtered_classes)
            self.update_network_weight_classaware(filtered_classes)

            self.loop += 1

        print('Training Done!')

