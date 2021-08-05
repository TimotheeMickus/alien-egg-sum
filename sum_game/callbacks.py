import gc
import json
import pathlib
import pprint

import egg.core as core

import torch

class EarlyStop(Exception):
    pass

class EarlyStopperCallback(core.Callback):
    def __init__(
        self, tracked_metric="acc", optimum="max", patience=10, save_dir=pathlib.Path("best_model")
    ):
        self.patience = patience
        self.metric = tracked_metric
        self.comp = optimum
        self.best_val = None
        self.strikes = 0
        self.save_dir = save_dir
        if self.comp == 'max':
            self.is_better = lambda val: self.best_val < val
        else:
            self.is_better = lambda val: self.best_val > new

    def save_if_best(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        to_save = True
        if (self.save_dir / "best_results.txt").is_file():
            with open(self.save_dir / "best_results.txt", "r") as istr:
                prev_best = float(istr.read())
                to_save = not self.is_better(prev_best)
        if to_save:
            if self.trainer.distributed_context.is_distributed:
                game = self.trainer.game.module
            else:
                game = self.trainer.game
            save_items = (0, game.state_dict(), self.trainer.optimizer.state_dict(), None)
            torch.save(save_items, self.save_dir / "model.pt")
            with open(self.save_dir / "best_results.txt", "w") as ostr:
                print(self.best_val, file=ostr)
            print('\033[92m' + "Saved new best model" + '\033[0m')


    def on_validation_end(self, loss, logs, epoch):
        dump = dict(loss=loss)
        if self.metric in logs.aux:
            dump[self.metric] = logs.aux[self.metric].mean().item()
        if self.best_val is None:
            self.best_val = dump[self.metric]
            self.save_if_best()
        else:
            new_val = dump[self.metric]
            if self.is_better(new_val):
                print('\033[92m' + f"new val {new_val} is better than {self.best_val}" + '\033[0m')
                self.best_val = new_val
                self.save_if_best()
                self.strikes = 0
            else:
                print('\033[91m' + f"new val {new_val} is no better than {self.best_val}" + '\033[0m')
                self.strikes += 1
        if self.strikes >= self.patience:
            print('\033[91m' + f"No improvement on {self.metric} after {self.patience} epochs, stopping early." + '\033[0m')
            raise EarlyStop

class LengthCurriculum(core.Callback):
    def __init__(self, n_epochs=100, n_updates=7):
        max_len = core.get_opts().max_len
        self.curriculum = {
            (n_epochs * i) // n_updates : max((max_len * i) // n_updates, 1)
            for i in range(n_updates + 1)
        }
        print("Curriculum:")
        pprint.pprint(self.curriculum)


    def on_train_begin(self, trainer):
        super().on_train_begin(trainer)
        core.get_opts().max_len = self.curriculum[0]
        if self.trainer.distributed_context.is_distributed:
            sender = self.trainer.game.module.sender
        else:
            sender = self.trainer.game.sender
        sender.max_len = self.curriculum[0]

    def on_epoch_begin(self, epoch):
        if epoch in self.curriculum:
            prev_max_len = core.get_opts().max_len
            new_max_len = self.curriculum[epoch]
            print('\033[94m' + f"updating length: {prev_max_len} => {new_max_len}" + '\033[0m')
            core.get_opts().max_len = new_max_len
            if self.trainer.distributed_context.is_distributed:
                sender = self.trainer.game.module.sender
            else:
                sender = self.trainer.game.sender
            sender.max_len = new_max_len
