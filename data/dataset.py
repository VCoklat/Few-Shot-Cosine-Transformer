# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import json, os, random
from PIL import Image
from torch.utils.data import Dataset
#import torchvision.transforms as T
import torchvision.transforms as transforms

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def identity(x):
    """No-op target_transform (kept for API compatibility)."""
    return x


# ----------------------------------------------------------------------
# main dataset class
# ----------------------------------------------------------------------
class SubDataset(Dataset):
    """
    A minimal class-balanced dataset wrapper.

    Parameters
    ----------
    sub_meta : dict
        Mapping {class_name: [img_path, …]} for a single split
        (base / val / test).
    cl : list[str]
        List of class names to keep in this SubDataset.
    transform : callable
        Image transform pipeline (default: transforms.ToTensor()).
    target_transform : callable
        Optional transform applied to the integer label.
    """
    def __init__(
        self,
        sub_meta: dict,
        cl: list[str],
        transform=transforms.ToTensor(),
        target_transform=identity,
    ):
        self.transform = transform
        self.target_transform = target_transform

        # ------------------------------------------------------------------
        # build flat sample list  →  [(img_path, cls_name), …]
        # ------------------------------------------------------------------
        self.samples = [
            (img_path, cls_name)
            for cls_name in cl
            for img_path in sub_meta[cls_name]
        ]

        # keep class names in a deterministic order for printing / mapping
        self.class_labels = sorted(cl)

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
