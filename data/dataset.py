# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
#import torchvision.transforms as T
import torchvision.transforms as transforms
import json, os
from PIL import Image
import torchvision.transforms as transforms       # NEW
from torch.utils.data import Dataset              # (already there if you copied earlier)

def identity(x):                                  # NEW
    return x

class SubDataset(Dataset):
    def __init__(self,
                 sub_meta: dict,
                 cl: list[str],
                 transform=transforms.ToTensor(),
                 target_transform=identity):      # ← uses identity
        self.transform = transform
        self.target_transform = target_transform

        # flat list: (img_path , cls_name)
        self.samples = [(p, c) for c in cl for p in sub_meta[c]]

        # readable names made available downstream
        self.class_labels = sorted(cl)            # NEW


    # --------------------- torch Dataset API -----------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_name = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # integer label 0 … (n_classes-1)
        label = self.class_labels.index(cls_name)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self,i):
        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed)
        
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)
        
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
