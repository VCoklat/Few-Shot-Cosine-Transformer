# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import json, os, random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class SetDataset(Dataset):
    """
    • Expects `data_file` to be a JSON mapping each class name to
      a list of image paths, e.g.
        {
          "cat"   : ["cat/0001.jpg", …],
          "dog"   : ["dog/0007.jpg", …],
          …
        }
    • `class_labels` (list[str]) is exposed so downstream code can show
      readable per-class metrics.
    """

    def __init__(self, data_file: str, transform: T.Compose | None):
        super().__init__()
        with open(data_file, "r") as f:
            self.meta = json.load(f)                # {cls: [paths]}
        self.class_labels = sorted(self.meta.keys()) # <-- NEW

        self.transform = transform
        # flatten into [(img_path, cls_idx), …]
        self.samples = []
        for cls_idx, cls_name in enumerate(self.class_labels):
            for path in self.meta[cls_name]:
                self.samples.append((path, cls_idx))

    # ------------- standard Dataset API ----------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
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
