"""
src/dataset.py
CelebA dataset loader using torchvision.

torchvision downloads CelebA automatically on first run (~1.4 GB).
If the Google Drive quota error appears, use the fallback cell
in the Colab notebook.

Splits (torchvision defaults):
    train  162,770 images
    valid   19,867 images
    test    19,962 images

Attributes are converted from {0, 1} to {-1, +1} for bipolar conditioning.
"""

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import CelebA


class CelebAAttrDataset(Dataset):
    """
    Wraps torchvision CelebA.
    Returns (image_tensor, attr_tensor):
        image_tensor  FloatTensor (3, IMG_SIZE, IMG_SIZE)  in [-1, 1]
        attr_tensor   FloatTensor (N_ATTRS,)               in {-1, +1}
    """

    def __init__(self, root, split: str, attr_names: list,
                 img_size: int, download: bool = True):
        transform = T.Compose([
            T.CenterCrop(178),
            T.Resize(img_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self._ds  = CelebA(root=str(root), split=split,
                           target_type="attr", transform=transform,
                           download=download)
        all_names = self._ds.attr_names
        self._idx = [all_names.index(a) for a in attr_names]

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, i):
        img, attrs = self._ds[i]
        sel = attrs[self._idx].float()
        sel = sel * 2 - 1          # {0,1} -> {-1,+1}
        return img, sel


def get_loaders(cfg):
    """
    Returns (train_loader, test_loader) built from Config.
    Downloads CelebA to cfg.DATA_DIR on first call.
    """
    kw = dict(root=cfg.DATA_DIR, attr_names=cfg.ATTRS,
              img_size=cfg.IMG_SIZE, download=True)
    train_ds = CelebAAttrDataset(split="train", **kw)
    test_ds  = CelebAAttrDataset(split="test",  **kw)

    lkw = dict(num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  **lkw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, **lkw)

    print(f"[dataset] train={len(train_ds):,}  test={len(test_ds):,}")
    return train_loader, test_loader
