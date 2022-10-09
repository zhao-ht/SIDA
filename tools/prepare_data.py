import os
import data.utils as data_utils
from data.custom_dataset_dataloader import CustomDatasetDataLoader
from data.class_aware_dataset_dataloader import ClassAwareDataLoader
from config.config import cfg
from data.image_folder import make_dataset_with_labels, make_dataset_classwise
from torch.utils.data import Dataset
import torch
import random
from math import ceil
from PIL import Image

def collate_fn(data):
    # data is a list: index indicates classes
    data_collate = {}
    num_classes = len(data)
    keys = data[0].keys()
    for key in keys:
        if key.find('Label') != -1:
            data_collate[key] = [torch.tensor(data[i][key]) for i in range(num_classes)]
        if key.find('Img') != -1:
            data_collate[key] = [data[i][key] for i in range(num_classes)]
        if key.find('Path') != -1:
            data_collate[key] = [data[i][key] for i in range(num_classes)]
        if key.find('Weight') != -1:
            data_collate[key] = [data[i][key] for i in range(num_classes)]

    return data_collate

class CategoricalSTDataset_important_weight(Dataset):
    def __init__(self):
        super(CategoricalSTDataset_important_weight, self).__init__()

    def initialize(self, source_root, target_paths,weight,
                   classnames, class_set,
                   source_batch_size,
                   target_batch_size, seed=None,
                   transform=None, **kwargs):
        assert weight.shape[1]==len(class_set)

        self.source_root = source_root
        self.target_paths = target_paths

        self.transform = transform
        self.class_set = class_set

        self.data_paths = {}
        self.data_paths['source'] = {}
        cid = 0
        for c in self.class_set:
            self.data_paths['source'][cid] = make_dataset_classwise(self.source_root, c)
            cid += 1

        self.data_paths['target'] = self.target_paths

        self.weight = weight

        self.seed = seed
        self.classnames = classnames

        self.batch_size = source_batch_size
        self.target_batch_size = target_batch_size


    def __getitem__(self, index):
        data = {}
        d='source'
        cur_paths = self.data_paths[d]
        if self.seed is not None:
            random.seed(self.seed)

        inds = random.sample(range(len(cur_paths[index])), \
                             self.batch_size)

        path = [cur_paths[index][ind] for ind in inds]
        data['Path_' + d] = path
        assert (len(path) > 0)
        for p in path:
            img = Image.open(p).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            if 'Img_' + d not in data:
                data['Img_' + d] = [img]
            else:
                data['Img_' + d] += [img]

        data['Label_' + d] = [self.classnames.index(self.class_set[index])] * len(data['Img_' + d])
        data['Img_' + d] = torch.stack(data['Img_' + d], dim=0)

        data['Weight_' + d] = [torch.ones(len(self.classnames)) for ind in inds]


        d='target'
        cur_paths = self.data_paths[d]
        if self.seed is not None:
            random.seed(self.seed)

        sampler = torch.utils.data.WeightedRandomSampler \
            (self.weight[:,index], self.batch_size, replacement=True)
        inds = list(sampler)
        path = [cur_paths[ind] for ind in inds]
        data['Path_' + d] = path
        assert (len(path) > 0)
        for p in path:
            img = Image.open(p).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            if 'Img_' + d not in data:
                data['Img_' + d] = [img]
            else:
                data['Img_' + d] += [img]

        data['Label_' + d] = [self.classnames.index(self.class_set[index])] * len(data['Img_' + d])
        data['Img_' + d] = torch.stack(data['Img_' + d], dim=0)

        weight = [self.weight[ind,index] for ind in inds]
        data['Weight_'+d]=weight

        return data

    def __len__(self):
        return len(self.class_set)

    def name(self):
        return 'CategoricalSTDataset_important_weight'

class ClassAwareDataLoader_weight(object):
    def name(self):
        return 'ClassAwareDataLoader_weight'

    def __init__(self,important_weight, source_batch_size, target_batch_size,
                 source_dataset_root="", target_paths=[],
                 transform=None,weight=None, classnames=[],
                 class_set=[], num_selected_classes=0,
                 seed=None, num_workers=0, drop_last=True,
                 sampler='RandomSampler', **kwargs):
        # dataset type
        if important_weight:
            self.dataset=CategoricalSTDataset_important_weight()

        self.important_weight=important_weight
        # dataset parameters
        self.source_dataset_root = source_dataset_root
        self.target_paths = target_paths
        self.classnames = classnames
        self.class_set = class_set
        self.source_batch_size = source_batch_size
        self.target_batch_size = target_batch_size
        self.seed = seed
        self.transform = transform
        self.weight=weight

        # loader parameters
        self.num_selected_classes = min(num_selected_classes, len(class_set))
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.sampler = sampler
        self.kwargs = kwargs

    def construct(self):
        self.dataset.initialize(source_root=self.source_dataset_root,
                                target_paths=self.target_paths,weight=self.weight,
                                classnames=self.classnames, class_set=self.class_set,
                                source_batch_size=self.source_batch_size,
                                target_batch_size=self.target_batch_size,
                                seed=self.seed, transform=self.transform,
                                **self.kwargs)

        drop_last = self.drop_last
        sampler = getattr(torch.utils.data, self.sampler)(self.dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                      self.num_selected_classes, drop_last)

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_sampler=batch_sampler,
                                                      collate_fn=collate_fn,
                                                      num_workers=int(self.num_workers))

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        if self.important_weight:
            dataset_len = 0.0
            cid = 0
            for c in self.class_set:
                c_len = len(self.dataset.data_paths['source'][cid]) // \
                             self.dataset.batch_size
                dataset_len += c_len
                cid += 1
            dataset_len = ceil(1.0 * dataset_len / self.num_selected_classes)
            dataset_len=max([dataset_len,len(self.dataset.data_paths['target'])/self.dataset.batch_size/self.num_selected_classes])
        else:
            dataset_len = 0.0
            cid = 0
            for c in self.class_set:
                c_len = max([len(self.dataset.data_paths[d][cid]) // \
                             self.dataset.batch_sizes[d][cid] for d in ['source', 'target']])
                dataset_len += c_len
                cid += 1

            dataset_len = ceil(1.0 * dataset_len / self.num_selected_classes)
            return dataset_len
        return int(dataset_len)

def prepare_data_SIDA():
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    # for clustering
    batch_size = cfg.CLUSTERING.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building clustering_%s dataloader...' % source)
    dataloaders['clustering_' + source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=False, num_workers=cfg.NUM_WORKERS, 
                classnames=classes)

    batch_size = cfg.CLUSTERING.TARGET_BATCH_SIZE
    dataset_type = cfg.CLUSTERING.TARGET_DATASET_TYPE 
    print('Building clustering_%s dataloader...' % target)
    dataloaders['clustering_' + target] = CustomDatasetDataLoader(
                dataset_root=dataroot_T, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=False, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    # class-agnostic source dataloader for supervised training
    batch_size = cfg.TRAIN.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building %s dataloader...' % source)
    dataloaders[source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=True, num_workers=cfg.NUM_WORKERS, 
                classnames=classes)

    # initialize the categorical dataloader
    dataset_type = 'CategoricalSTDataset'
    source_batch_size = cfg.TRAIN.SOURCE_CLASS_BATCH_SIZE
    target_batch_size = cfg.TRAIN.TARGET_CLASS_BATCH_SIZE
    print('Building categorical dataloader...')
    dataloaders['categorical'] = ClassAwareDataLoader(
                dataset_type=dataset_type, 
                source_batch_size=source_batch_size, 
                target_batch_size=target_batch_size, 
                source_dataset_root=dataroot_S, 
                transform=train_transform, 
                classnames=classes, 
                num_workers=cfg.NUM_WORKERS,
                drop_last=True, sampler='RandomSampler')


    dataset_type = 'CategoricalSTDataset'
    source_batch_size = cfg.TRAIN.SOURCE_CLASS_BATCH_SIZE
    target_batch_size = cfg.TRAIN.TARGET_CLASS_BATCH_SIZE
    print('Building categorical dataloader...')
    dataloaders['categorical_weight'] = ClassAwareDataLoader_weight(
        important_weight=cfg.FIXED.IMPORTANTWEIGHT,
                dataset_type=dataset_type,
                source_batch_size=source_batch_size,
                target_batch_size=target_batch_size,
                source_dataset_root=dataroot_S,
                transform=train_transform,
                classnames=classes,
                num_workers=cfg.NUM_WORKERS,
                drop_last=True, sampler='RandomSampler')





    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    dataloaders['test'] = CustomDatasetDataLoader(
                    dataset_root=dataroot_test, dataset_type=dataset_type,
                    batch_size=batch_size, transform=test_transform,
                    train=False, num_workers=cfg.NUM_WORKERS,
                    classnames=classes)

    return dataloaders

