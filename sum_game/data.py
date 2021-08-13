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
    """convert dataset to dataloader"""
    return DataLoader(
        dataset,
        batch_sampler=SumGameSampler(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        ),
    )


class SumGameSampler(Sampler):
    """A (probably useless) sampler"""

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


class SumGameOneHotDataset(Dataset):
    """Represent integers as one-hot vectors"""

    def __init__(self, path, N=None, keep_pairs=False):
        self.N = N
        self.two_N = 2 * N
        self.one_hots = torch.eye(N + 1)
        self.one_hots.requires_grad = False
        self.keep_pairs = False
        self.items = []
        with open(path, "r") as istr:
            istr = map(str.strip, istr)
            istr = map(str.split, istr)
            for row in istr:
                row = list(map(int, row))
                self.items.append(row)
        print(f"N={self.N}")

    def get_n_features(self):
        return self.one_hots.size(0) * 2

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        a, b, s = self.items[idx]
        ipt = torch.cat([self.one_hots[a], self.one_hots[b]], dim=0).float()
        # retrieve the one hots and the target
        if self.keep_pairs:
            return ipt, s, torch.tensor([a, b], dtype=torch.float)
        return ipt, s


class SumGameStructuredDataset(Dataset):
    """Represent integers through binary expansion"""

    def __init__(self, path, two_N, keep_pairs=False):
        self.two_N = two_N
        self.items = []
        self.keep_pairs = keep_pairs
        with open(path, "r") as istr:
            istr = map(str.strip, istr)
            istr = map(str.split, istr)
            for row in istr:
                row = list(map(int, row))
                self.items.append(row)
        self.bitlen = len(bin(self.two_N)[2:])
        print(f"2N={self.two_N}, bitlen={self.bitlen}")

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
        if self.keep_pairs:
            return (
                torch.tensor(ipt, dtype=torch.float),
                s,
                torch.tensor([a, b], dtype=torch.float),
            )
        # retrieve the bitvecs and the target
        return torch.tensor(ipt, dtype=torch.float), s


def generate_datafiles(data_dir, N):
    """Simple function to generate data, based on the max value N of integers to sum"""
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
