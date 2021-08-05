import functools
import gc
import json
import mmap
import pathlib
import random

import egg.core as core
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


def to_dataloader(dataset, batch_size, shuffle=True, drop_last=False):
    return DataLoader(
        dataset,
        batch_sampler=SumGameSampler(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        ),
    )


class SumGameSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._len = len(self.dataset) // batch_size
        if len(self.dataset) % batch_size != 0:
            self._len += 1

    def __len__(self):
        return self._len

    def __iter__(self):
        indices = range(len(self.dataset))
        if self.shuffle:
            indices = list(indices)
            random.shuffle(indices)
        acc = []
        for idx in indices:
            acc.append(idx)
            if len(acc) == self.batch_size:
                yield acc
                acc = []
        if acc and not self.drop_last:
            yield acc


class SumGameDataset(Dataset):
    """Working out a memory efficient dataset representation
    Here: N is the maximum integer for the operator and operand in our sum game, so the total is at most 2N"""

    def __init__(self, path, maxint=0):
        self.maxint = maxint
        self.items = []
        with open(path, "r") as istr:
            istr = map(str.strip, istr)
            istr = map(str.split, istr)
            for row in istr:
                row = list(map(int, row))
                self.items.append(row)
                self.maxint = max(self.maxint, row[-1])
        self.bitlen = len(bin(self.maxint)[2:])
        print(f"maxint={self.maxint}, bitlen={self.bitlen}")

    def get_n_features(self):
        return self.bitlen * 2

    @functools.lru_cache(512)
    def get_bit_rep(self, intval):
        pad = [0] * self.bitlen
        bitfield = [1 if d == "1" else 0 for d in bin(intval)[2:]]
        return (pad + bitfield)[-self.bitlen :]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        a, b, s = self.items[idx]
        ipt = self.get_bit_rep(a) + self.get_bit_rep(b)
        # retrieve the one hots and the target
        return torch.tensor(ipt, dtype=torch.float), s


def generate_datafiles(data_dir, N):
    """Simple function to generate data, based on the max value N of the operand and operator"""
    data_dir = pathlib.Path(data_dir)
    assert data_dir.is_dir()
    all_items = [(a, b, a + b) for a in range(N) for b in range(N)]
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
