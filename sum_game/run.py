# intended main entry point.

import egg.core as core

from config import *
from data import *
from archs import get_game

import torch.optim as optim
from torch.utils.data import DataLoader
import torch
torch.autograd.set_detect_anomaly(True)

if generate_data:
    print(f"generating data in {data_dir}...")
    generate_datafiles()
    print("done.")

print("building data loaders...")
train_loader = SumGameDataLoader(data_dir / "train.txt")
dev_loader = SumGameDataLoader(data_dir / "dev.txt")
print("done.")

print("building trainer...")
game, callbacks = get_game()
optimizer = core.build_optimizer(game.parameters())
trainer = core.Trainer(
    game=game,
    optimizer=optimizer,
    train_data=train_loader,
    validation_data=dev_loader,
    callbacks=callbacks
    + [
        core.ConsoleLogger(print_train_loss=True, as_json=True),
        # core.PrintValidationEvents(n_epochs=n_epochs),
    ],
)
print(f"done.\nGame: {game}")

print("training...")
trainer.train(n_epochs=n_epochs)
print("done.")
