# This file handles everything regarding data: synthetic data, datasets and dataloaders

import pathlib
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import *

def as_dataloader(clazz):
    """This is a silly decorator that saves me from wrapping the dataset in a dataloader in run.py"""
    def _wrap(*args, batch_size=batch_size, **kwargs):
        dataset = clazz(*args, **kwargs)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
        )
    return _wrap

class SumGameDataset(Dataset):
    """adapted from https://github.com/facebookresearch/EGG/blob/master/egg/zoo/basic_games/data_readers.py
    Here: N is the maximum integer for the operator and operand in our sum game, so the total is at most 2N"""
    def __init__(self, path, N=maxint):
        self.frame = []
        with open(path, "r") as istr:
            for row in istr:
                row = list(map(int, row.split()))
                operator, operand, result = row
                z = torch.zeros((2, N))
                z[0, operator] = 1
                z[1, operand] = 1
                y = torch.tensor(result, dtype=torch.long)
                self.frame.append((z.view(-1), y))

    def get_n_features(self):
        return self.frame[0][0].size(0)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]

SumGameDataLoader = as_dataloader(SumGameDataset)

def generate_datafiles(data_dir=data_dir, N=maxint):
    """Simple function to generate data, based on the max value N of the operand and operator"""
    data_dir = pathlib.Path(data_dir)
    assert data_dir.is_dir()
    all_items = [(a, b, a+b) for a in range(N) for b in range(N)]
    random.shuffle(all_items)
    idx8 = len(all_items) * 8 // 10
    idx9 = len(all_items) * 9 // 10
    train = all_items[:idx8]
    dev = all_items[idx8:idx9]
    test = all_items[idx9:]
    with open(data_dir / "train.txt", "w") as ostr:
        for item in train:
            print(*item, file=ostr)
    with open(data_dir / "dev.txt", "w") as ostr:
        for item in dev:
            print(*item, file=ostr)
    with open(data_dir / "test.txt", "w") as ostr:
        for item in test:
            print(*item, file=ostr)
