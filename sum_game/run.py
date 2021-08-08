# intended main entry point.
import math
import pathlib
import pickle
import secrets
import sys

import egg.core as core

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import skopt

from callbacks import EarlyStopperCallback, EarlyStop, LengthCurriculum
from config import get_args, get_search_space
from data import (
    SumGameStructuredDataset,
    SumGameOneHotDataset,
    to_dataloader,
    generate_datafiles,
)
from archs import get_game

# torch.autograd.set_detect_anomaly(True)


def train_model(
    train_loader,
    dev_loader,
    game_type="categorization",
    batch_size=32,
    n_hidden=10,
    embed_dim=10,
    max_len=10,
    vocab_size=10,
    cell="rnn",
    entropy_coeff=1e-1,
    tensorboard_dir=None,
    device=torch.device("cuda:0"),
    n_epochs=10,
    use_curriculum=False,
    curriculum_length=20,
    n_updates=7,
    save_dir=pathlib.Path("best_model"),
    mechanism="reinforce",
    temperature=1.0,
    reduction="none",
    **ignore,
):

    early_stopper = EarlyStopperCallback(save_dir=save_dir)

    game, cbs = get_game(
        game_type=game_type,
        n_features=train_loader.dataset.get_n_features(),
        maxint=train_loader.dataset.two_N,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        n_hidden=n_hidden,
        cell=cell,
        max_len=max_len,
        entropy_coeff=entropy_coeff,
        temperature=temperature,
        mechanism=mechanism,
        reduction=reduction,
    )
    optimizer = core.build_optimizer(game.parameters())
    if tensorboard_dir:
        path = pathlib.Path(tensorboard_dir) / secrets.token_urlsafe(8)
        cbs += [core.callbacks.TensorboardLogger(writer=SummaryWriter(path))]
    if use_curriculum:
        cbs += [
            LengthCurriculum(n_updates=n_updates, n_epochs=curriculum_length),
        ]
    trainer = core.Trainer(
        game=game,
        device=device,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=dev_loader,
        callbacks=cbs
        + [
            core.callbacks.ConsoleLogger(print_train_loss=True, as_json=True),
            early_stopper
            # core.ProgressBarLogger(
            #     n_epochs=args.n_epochs,
            #     train_data_len=math.ceil(len(train) / batch_size),
            #     test_data_len=math.ceil(len(dev) / batch_size),
            #     use_info_table=False,
            # )
            # core.PrintValidationEvents(n_epochs=n_epochs),
        ],
    )
    print(f"Game: {game}")

    print("training...")
    try:
        trainer.train(n_epochs=n_epochs)
    except EarlyStop:
        pass
    # right now I'm hacking this together, so 🙈
    core.get_opts().max_len = max_len
    return trainer, early_stopper.best_val


if __name__ == "__main__":
    args = get_args()
    if args.do_generate_data:
        print(f"generating data in {args.data_dir}...")
        generate_datafiles(args.data_dir, args.maxint)
        print("done.")
    train_dss = {
        "structured": SumGameStructuredDataset(
            args.data_dir / "train.txt", two_N=(args.maxint - 1) * 2
        ),
        "one-hot": SumGameOneHotDataset(
            args.data_dir / "train.txt", N=(args.maxint - 1)
        ),
    }
    dev_dss = {
        "structured": SumGameStructuredDataset(
            args.data_dir / "dev.txt", two_N=(args.maxint - 1) * 2
        ),
        "one-hot": SumGameOneHotDataset(args.data_dir / "dev.txt", N=(args.maxint - 1)),
    }

    if args.do_hypertune:
        if args.gp_result_dump.is_file():
            print(
                f"will not override existing file {args.gp_result_dump}, stopping instead."
            )
            sys.exit(0)
        else:
            open(args.gp_result_dump, "w").close()
        search_space = get_search_space()
        print(f"searching for optimal hyperparameters.")

        @skopt.utils.use_named_args(search_space)
        def gp_train(**hparams):
            torch.cuda.empty_cache()
            hparams["embed_dim"] = 2 ** hparams["embed_pow"]
            hparams["n_hidden"] = 2 ** hparams["hidden_pow"]
            hparams["batch_size"] = 2 ** hparams["batch_pow"]
            print(f"hparams: {hparams}")
            core.get_opts().__dict__.update(hparams)
            print("building data loaders...")
            train_ds = train_dss[hparams["ipt_format"]]
            dev_ds = dev_dss[hparams["ipt_format"]]
            train_loader = to_dataloader(
                train_ds, batch_size=int(hparams["batch_size"])
            )
            dev_loader = to_dataloader(dev_ds, batch_size=1024)
            print("done.")

            best_per_run = []
            for run in range(args.runs_per_conf):
                all_kwargs = vars(args)
                all_kwargs.update(hparams)
                trainer, best_val = train_model(train_loader, dev_loader, **all_kwargs)
                del trainer
                torch.cuda.empty_cache()
                print(f"best score for run: {best_val}.")
                best_per_run.append(best_val)
            print(
                f"training of all {args.runs_per_conf} was completed. "
                + f"Overall best loss: {max(best_per_run)}, "
                + f"average best: {sum(best_per_run) / args.runs_per_conf}"
            )

            # minus sign for gp minimize
            return -sum(best_per_run) / args.runs_per_conf

        hp_search_result = skopt.gp_minimize(gp_train, search_space)
        print(
            f"search for optimal hyperparameters is done. results: {hp_search_result}"
        )
        with open(args.gp_result_dump, "wb") as ostr:
            pickle.dump(hp_search_result, ostr)
    else:
        print("default training behavior.")
        train_loader = to_dataloader(train, batch_size=args.batch_size)
        dev_loader = to_dataloader(dev, batch_size=args.batch_size)
        trainer, _ = train_model(train_loader, dev_loader, **vars(args))
