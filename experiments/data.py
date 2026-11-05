"""CIFAR-100 数据加载 (在 training host 上下载, seed=42)。"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def get_cifar100_loaders(data_dir, batch_size=256, num_workers=4, val_split=5000, seed=42):
    """返回 (train_loader, val_loader, test_loader)。
    从官方 train 划出 val_split 作为验证集 (选超参用); test 仅最终画图。
    """
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    full_train = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=train_tf)
    full_train_eval = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=eval_tf)
    test_set = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=eval_tf)

    n = len(full_train)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    val_idx = perm[:val_split].tolist()
    train_idx = perm[val_split:].tolist()

    train_sub = torch.utils.data.Subset(full_train, train_idx)
    val_sub = torch.utils.data.Subset(full_train_eval, val_idx)  # val 用 eval transform

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_sub, batch_size=512, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
