# This file defines the game and agents. Mostly inspired from the basic recognition game on egg

import egg.core as core

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

class Sender(nn.Module):
    def __init__(self, n_hidden=n_hidden, n_features=n_features):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _aux_input):
        return self.fc1(x)


class Receiver(nn.Module):
    def __init__(self, n_hidden=n_hidden, n_features=n_features):
        super(Receiver, self).__init__()
        self.output = nn.Linear(n_hidden, n_features)

    def forward(self, x, _input, _aux_input):
        return self.output(x)

def get_game(sender=Sender(), receiver=Receiver()):
    def loss_fn(sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
        batch_size = sender_input.size(0)
        labels = labels
        receiver_guesses = receiver_output.argmax(dim=1).detach()
        acc = (receiver_guesses == labels).float().mean().detach()
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        loss = loss.view(batch_size, -1).mean(dim=1)
        return loss, {"acc": acc.detach()}

    sender = core.RnnSenderGS(
        sender,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=n_hidden,
        cell=cell,
        max_len=max_len,
        temperature=temperature,
    )
    receiver = core.RnnReceiverGS(
        receiver,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=n_hidden,
        cell=cell,
    )
    game = core.SenderReceiverRnnGS(sender, receiver, loss_fn)
    callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)]
    return game, callbacks
