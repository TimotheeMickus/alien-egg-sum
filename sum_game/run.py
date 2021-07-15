# intended main entry point.
import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
import math
import egg.core as core

from config import *
from data import *
from archs import *

import torch.optim as optim
from torch.utils.data import DataLoader
# torch.autograd.set_detect_anomaly(True)
import skopt

import pickle

if __name__ == "__main__":
    if generate_data:
        print(f"generating data in {data_dir}...")
        generate_datafiles()
        print("done.")

    train, dev = SumGameDataset(data_dir / "train.txt"), SumGameDataset(data_dir / "dev.txt")

    search_space = [
        skopt.space.Integer(1, 8, "uniform", name="embed_pow"), # window size
        skopt.space.Integer(1, 8, "uniform", name="hidden_pow"), # number of negative items
        skopt.space.Integer(4, 10, "uniform", name="batch_pow"), # number of negative items
        skopt.space.Integer(1, 5, "uniform", name="max_len"), # negative sampling items selection
        skopt.space.Integer(5, 500, "log-uniform", name="vocab_size"), # learning rate
        skopt.space.Real(1e-6, 0.1, "log-uniform", name="lr"), # high-freq words temperature
        skopt.space.Real(0.25, 1.75, "uniform", name="temperature"), # smallest value to learning rate
        skopt.space.Categorical(["rnn", "gru", "lstm"], name="cell")
    ]

    @skopt.utils.use_named_args(search_space)
    def train_model(**hparams):
        torch.cuda.empty_cache()
        hparams["embed_dim"] = 2 ** hparams["embed_pow"]
        hparams["n_hidden"] = 2 ** hparams["hidden_pow"]
        hparams["batch_size"] = 2 ** hparams["batch_pow"]
        print(f"hparams: {hparams}")
        core.get_opts().__dict__.update(hparams)
        globals().update(hparams)
        print("building data loaders...")
        train_loader = to_dataloader(train, batch_size=int(batch_size))
        dev_loader = to_dataloader(dev, batch_size=int(batch_size))
        print("done.")

        best_per_run = []
        for run in range(runs_per_conf):
            tracker = CustomTracker(print_train_loss=True, as_json=True, tracked_metric="acc", optimum="max")
            print(f"run #{run}, building trainer...")
            sender, receiver = Sender(n_hidden=n_hidden), Receiver(n_hidden=n_hidden)
            game, callbacks = get_game(
                sender=sender,
                receiver=receiver,
                n_hidden=n_hidden,
                embed_dim=embed_dim,
                max_len=max_len,
                vocab_size=vocab_size,
                cell=cell,
                temperature=temperature,
            )
            optimizer = core.build_optimizer(game.parameters())
            trainer = core.Trainer(
                game=game,
                device=device,
                optimizer=optimizer,
                train_data=train_loader,
                validation_data=dev_loader,
                callbacks=callbacks
                + [
                    tracker,
                    core.ProgressBarLogger(
                        n_epochs=n_epochs,
                        train_data_len=math.ceil(len(train) / batch_size),
                        test_data_len=math.ceil(len(dev) / batch_size),
                        use_info_table=False
                    )
                    # core.PrintValidationEvents(n_epochs=n_epochs),
                ],
            )
            print(f"done.\nGame: {game}")

            print("training...")
            trainer.train(n_epochs=n_epochs)
            print(f"done, best score for run: {tracker.best_val}.")
            best_per_run.append(tracker.best_val)
        print(f"training of all {runs_per_conf} was completed. Overall best acc: {max(best_per_run)}, average best: {sum(best_per_run) / runs_per_conf}")
        return -sum(best_per_run)/runs_per_conf # minus sign for gp minimize

    print("searching for optimal hyperparameters")
    hp_search_result = skopt.gp_minimize(train_model, search_space)
    print(f"search for optimal hyperparameters is done. results: {hp_search_result}")
    with open("skopt_gp.pkl", "wb") as ostr:
        pickle.dump(hp_search_result, ostr)
