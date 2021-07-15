# This file handles everything regarding data: reporting, synthetic data, datasets and dataloaders
import gc
import json
import mmap
import pathlib
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import *

def to_dataloader(dataset, batch_size=batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

def as_dataloader(clazz):
    """This is a silly decorator that saved me from wrapping the dataset in a dataloader in run.py"""
    def _wrap(*args, batch_size=batch_size, **kwargs):
        dataset = clazz(*args, **kwargs)
        return to_dataloader(dataset, batch_size=batch_size)
    return _wrap

class SumGameDataset(Dataset):
    """Working out a memory efficient dataset representation
    Here: N is the maximum integer for the operator and operand in our sum game, so the total is at most 2N"""
    def __init__(self, path, N=maxint):
        self.items = torch.eye(N) # creates N one-hot vectors
        self.items.requires_grad = False
        self._lines = [0]
        self._file = open(path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, prot=mmap.PROT_READ)
        for line in iter(self._mm.readline, b""):
            self._lines.append(self._mm.tell())
        # the last line is empty
        self._lines = self._lines[:-1]

    def get_n_features(self):
        return self.items[0].size(0)

    def __len__(self):
        return len(self._lines)

    def __getitem__(self, idx):
        # goto line index
        self._mm.seek(self._lines[idx])
        # retrieve line
        line = self._mm.readline()
        # parse it
        a, b, s = map(int, line.decode().split())
        # retrieve the one hots and the target
        return torch.cat([self.items[a], self.items[b]], dim=0), s

SumGameDataLoader = as_dataloader(SumGameDataset)

class CustomTracker(core.Callback):
    def __init__(self, print_train_loss=False, as_json=False, tracked_metric="acc", optimum="max"):
        self.print_train_loss = print_train_loss
        self.as_json = as_json
        self.metric = tracked_metric
        self.comp = max if optimum == "max" else min
        self.best_val = None

    def aggregate_print(self, loss: float, logs: core.Interaction, mode: str, epoch: int):
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)

        if self.as_json:
            dump.update(dict(mode=mode, epoch=epoch))
            output_message = json.dumps(dump)
        else:
            output_message = ", ".join(sorted([f"{k}={v}" for k, v in dump.items()]))
            output_message = f"{mode}: epoch {epoch}, loss {loss}, " + output_message
        print(output_message, flush=True)

    def on_validation_end(self, loss: float, logs: core.Interaction, epoch: int):
        self.aggregate_print(loss, logs, "test", epoch)
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)
        if self.best_val is None:
            self.best_val = dump[self.metric]
        else:
            self.best_val = self.comp(dump[self.metric], self.best_val)
        torch.cuda.empty_cache()
        gc.collect()

    def on_epoch_end(self, loss: float, logs: core.Interaction, epoch: int):
        if self.print_train_loss:
            self.aggregate_print(loss, logs, "train", epoch)
        # torch.cuda.empty_cache()
        # gc.collect()

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
