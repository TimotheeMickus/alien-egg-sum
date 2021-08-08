import egg.core as core

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sender(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _aux_input):
        return self.fc1(x)


class ReceiverRegression(nn.Module):
    def __init__(self, n_hidden, maxint, n_features=1):
        super(ReceiverRegression, self).__init__()
        self.output = nn.Linear(n_hidden, n_features)
        self.register_buffer("maxint", torch.tensor(maxint))

    def forward(self, x, _input, _aux_input):
        return torch.sigmoid(self.output(x)) * self.maxint


class ReceiverCategorization(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(ReceiverCategorization, self).__init__()
        self.output = nn.Linear(n_hidden, n_features)

    def forward(self, x, _input, _aux_input):
        return self.output(F.relu(x))


class CategorizationLoss(nn.Module):
    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
        _aux_input,
    ):
        redux = self.reduction if self.training else "none"
        receiver_guess = receiver_output.argmax(dim=1)
        acc = (receiver_guess == labels.view_as(receiver_guess)).float().detach()
        loss = F.cross_entropy(receiver_output, labels.view(-1), reduction=redux)
        if redux != "none":
            loss = loss.view(1).expand_as(labels)
        return loss, {"acc": acc, "bare_loss": loss.detach()}


class RegressionLoss(nn.Module):
    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
        _aux_input,
    ):
        redux = self.reduction if self.training else "none"
        receiver_output = receiver_output.view(-1)
        labels = labels.view(-1)
        acc = (
            (receiver_output.round().int() == labels)
            .float()
            .detach()
            .view_as(receiver_output)
        )
        loss = F.mse_loss(receiver_output, labels.float(), reduction=redux)
        if redux != "none":
            loss = loss.view(1).expand_as(labels)
        return loss, {"acc": acc, "bare_loss": loss.detach()}


def get_game(
    game_type,
    n_features,
    maxint,
    vocab_size,
    embed_dim,
    n_hidden,
    cell,
    max_len,
    entropy_coeff,
    temperature,
    mechanism,
    reduction,
):
    sender = Sender(n_hidden=n_hidden, n_features=n_features)
    if game_type == "regression":
        receiver = ReceiverRegression(n_hidden=n_hidden, maxint=maxint, n_features=1)
        loss = RegressionLoss(reduction=reduction)
    else:
        receiver = ReceiverCategorization(n_hidden=n_hidden, n_features=maxint + 1)
        loss = CategorizationLoss(reduction=reduction)
    if mechanism == "reinforce":
        sender = core.RnnSenderReinforce(
            sender,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=n_hidden,
            cell=cell,
            max_len=max_len,
        )
        receiver = core.RnnReceiverDeterministic(
            receiver,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=n_hidden,
            cell=cell,
        )
        game = core.SenderReceiverRnnReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=entropy_coeff,
            receiver_entropy_coeff=0,
        )
        callbacks = []
    else:
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
        game = core.SenderReceiverRnnGS(
            sender,
            receiver,
            loss,
        )
        callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)]
    return game, callbacks
