# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import pdb
import torch
import torch.multiprocessing
from PIL import Image
import numpy as np
import random
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SetDataset, EpisodicBatchSampler
from abc import abstractmethod

# Set sharing strategy to file_descriptor to avoid "Trying to resize storage
# that is not resizable" error with multiprocessing DataLoader
torch.multiprocessing.set_sharing_strategy('file_descriptor')

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        if transform_type == 'RandomSizedCrop':
            # Use the new name instead
            return transforms.RandomResizedCrop(self.image_size)  # Changed self.size to self.image_size
        elif transform_type == 'CenterCrop':
            return transforms.CenterCrop(self.image_size)  # Changed self.size to self.image_size
        method = getattr(transforms, transform_type)
        
        # Rest of the method...
        # Note: This part of your code has duplication - you handle RandomSizedCrop 
        # twice, which is confusing. The second check won't ever be reached.
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type == 'Resize':
            return method(self.image_size)
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Normalize':
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
                transform_list = ['Resize','RandomSizedCrop', 'ColorJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']
            
        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 


def seed_worker(worker_id):
    worker_seed = worker_id #torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, k_shot, n_query, n_episode):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = k_shot + n_query
        self.n_episode = n_episode
        self.trans_loader = TransformLoader(image_size)

    # parameters that would change on train/val set
    def get_data_loader(self, data_file, aug):
        g = torch.Generator()
        g.manual_seed(0)
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(data_file, self.batch_size,
                             transform)
        sampler = EpisodicBatchSampler(
            len(dataset), self.n_way, self.n_episode)
        # Use persistent_workers=True to keep worker processes alive between
        # iterations, which helps avoid memory management issues with tensor storage.
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 8, pin_memory = True, worker_init_fn=seed_worker, generator=g, persistent_workers=True)     
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


