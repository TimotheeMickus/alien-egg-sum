# intended main entry point.

import egg.core as core

from config import *
from data import *
from archs import get_game

import torch.optim as optim
from torch.utils.data import DataLoader


if generate_data:
    generate_datafiles()

train_loader = SumGameDataLoader(data_dir / "train.txt")
dev_loader = SumGameDataLoader(data_dir / "dev.txt")
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
        core.PrintValidationEvents(n_epochs=n_epochs),
    ],
)

trainer.train(n_epochs=n_epochs)
