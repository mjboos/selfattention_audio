import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-7


def compute_attention_weights(logits: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Computes attention weights from logits by normalizing and transforming onto 0 to 1 interval.

    :param logits: tensor of logits of shape (batch_size, sequence_steps)
    :type logits: torch.Tensor
    :param eps: system error, defaults to 1e-7
    :type eps: float, optional
    :return: tensor of attention weights in interval (0,1) of dimension (batch_size, sequence_steps)
    :rtype: torch.Tensor
    """
    ai = torch.exp(logits - torch.max(logits, 1, keepdim=True)[0])
    att_weights = ai / (torch.sum(ai, dim=1, keepdim=True) + eps)
    return att_weights


class GRU_with_attention(nn.Module):
    """A Gated Recurring Unit (GRU) model with attention weighting applied to the last layer to compress all hidden states into a single tensor.
    This tensor is transformed to predict the target via `post_gru_seq` (which can be any torch Sequential object)."""

    def __init__(
        self,
        input_size: int = 31,
        hidden_size: int = 50,
        num_layers: int = 1,
        dropout: float = 0.0,
        n_targets: int = 2,
        bidirectional: bool = False,
        post_gru_seq: Optional[nn.Sequential] = None,
        batch_first: bool = True,
        **kwargs
    ):
        """
        :param input_size: size of the input tensor at each step, e.g. number of frequencies, defaults to 31
        :type input_size: int, optional
        :param hidden_size: hidden size to use for GRU, defaults to 50
        :type hidden_size: int, optional
        :param num_layers: number of GRU layers, defaults to 1
        :type num_layers: int, optional
        :param dropout: amount of dropout to apply, defaults to 0.0
        :type dropout: float, optional
        :param n_targets: number of targets to predict, defaults to 2
        :type n_targets: int, optional
        :param bidirectional: whether to run the GRU bidirectional, defaults to False
        :type bidirectional: bool, optional
        :param post_gru_seq: the layer(s) to apply to the tensor produced by attention weighting, None defaults to a linear transformation
        :type post_gru_seq: Optional[nn.Sequential], optional
        :param batch_first: whether the first dimension in the tensor is batch, defaults to True
        :type batch_first: bool, optional
        """
        super(GRU_with_attention, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )
        if post_gru_seq is None:
            self.post_gru = nn.Sequential(nn.Linear(hidden_size, n_targets))
        else:
            self.post_gru = post_gru_seq
        self.att = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x):
        x, hn = self.gru(x)
        # x is (batch, seq, hidden)
        logits = self.att(x)
        att_weights = compute_attention_weights(logits)
        x = (x * att_weights).sum(dim=1)
        x = self.post_gru(x)
        return x


class SimpleGRU(nn.Module):
    def __init__(
        self,
        input_size: int = 31,
        hidden_size: int = 50,
        num_layers: int = 1,
        dropout: float = 0.0,
        n_targets: int = 2,
        bidirectional: bool = False,
        post_gru_seq: Optional[nn.Sequential] = None,
        batch_first: bool = True,
        **kwargs
    ):
        """
        :param input_size: size of the input tensor at each step, e.g. number of frequencies, defaults to 31
        :type input_size: int, optional
        :param hidden_size: hidden size to use for GRU, defaults to 50
        :type hidden_size: int, optional
        :param num_layers: number of GRU layers, defaults to 1
        :type num_layers: int, optional
        :param dropout: amount of dropout to apply, defaults to 0.0
        :type dropout: float, optional
        :param n_targets: number of targets to predict, defaults to 2
        :type n_targets: int, optional
        :param bidirectional: whether to run the GRU bidirectional, defaults to False
        :type bidirectional: bool, optional
        :param post_gru_seq: the layer(s) to apply to the tensor produced by attention weighting, None defaults to a linear transformation
        :type post_gru_seq: Optional[nn.Sequential], optional
        :param batch_first: whether the first dimension in the tensor is batch, defaults to True
        :type batch_first: bool, optional
        """
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )
        if post_gru_seq is None:
            self.post_gru = nn.Sequential(nn.Linear(hidden_size, n_targets))
        else:
            self.post_gru = post_gru_seq

    def forward(self, x):
        x, hn = self.gru(x)
        x = self.post_gru(hn)
        return x


class GRU_with_concat_attention_per_target(nn.Module):
    """A Gated Recurring Unit (GRU) model with attention weighting to compress all hidden states into a single tensor.
    Hidden states are concatenated across layers and attention weighting applied to these concatenated states to produce a single tensor.
    This tensor is transformed to predict the target via `post_gru_seq` (which can be any torch Sequential object)."""

    def __init__(
        self,
        input_size: int = 31,
        hidden_size: int = 50,
        num_layers: int = 1,
        dropout: float = 0.0,
        n_targets: int = 2,
        bidirectional: bool = False,
        post_gru_seq: Optional[nn.Sequential] = None,
        batch_first: bool = True,
        **kwargs
    ):
        """
        :param input_size: size of the input tensor at each step, e.g. number of frequencies, defaults to 31
        :type input_size: int, optional
        :param hidden_size: hidden size to use for GRU, defaults to 50
        :type hidden_size: int, optional
        :param num_layers: number of GRU layers, defaults to 1
        :type num_layers: int, optional
        :param dropout: amount of dropout to apply, defaults to 0.0
        :type dropout: float, optional
        :param n_targets: number of targets to predict, defaults to 2
        :type n_targets: int, optional
        :param bidirectional: whether to run the GRU bidirectional, defaults to False
        :type bidirectional: bool, optional
        :param post_gru_seq: the layer(s) to apply to the tensor produced by attention weighting, None defaults to a linear transformation
        :type post_gru_seq: Optional[nn.Sequential], optional
        :param batch_first: whether the first dimension in the tensor is batch, defaults to True
        :type batch_first: bool, optional
        """
        super(GRU_with_concat_attention_per_target, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )
        self.att_layers = nn.ModuleList(
            [nn.Linear(hidden_size, 1, bias=True) for _ in range(n_targets)]
        )
        if post_gru_seq is None:
            self.post_gru = nn.Sequential(nn.Linear(n_targets * hidden_size, n_targets))
        else:
            self.post_gru = post_gru_seq

    def forward(self, x):
        x, hn = self.gru(x)
        # x is (batch, seq, hidden)
        x_tmp = []
        for att_lay in self.att_layers:
            logits = att_lay(x)
            att_weights = compute_attention_weights(logits)
            x_tmp.append((x * att_weights).sum(dim=1))
        x = self.post_gru(torch.cat(x_tmp, dim=-1))
        return x


class GRU_with_shared_attention_per_target(nn.Module):
    """A Gated Recurring Unit (GRU) model with attention weighting applied to the last layer to compress all hidden states into a single tensor.
    There are as many attention layers as there are targets, yet the resulting multiple compressed hidden states are shared (i.e. concatenated) to predict targets.
    This tensor is transformed to predict the target via `post_gru_seq` (which can be any torch Sequential object)."""

    def __init__(
        self,
        input_size: int = 31,
        hidden_size: int = 50,
        num_layers: int = 1,
        dropout: float = 0.0,
        n_targets: int = 2,
        bidirectional: bool = False,
        post_gru_seq: Optional[nn.Sequential] = None,
        batch_first: bool = True,
        **kwargs
    ):
        """
        :param input_size: size of the input tensor at each step, e.g. number of frequencies, defaults to 31
        :type input_size: int, optional
        :param hidden_size: hidden size to use for GRU, defaults to 50
        :type hidden_size: int, optional
        :param num_layers: number of GRU layers, defaults to 1
        :type num_layers: int, optional
        :param dropout: amount of dropout to apply, defaults to 0.0
        :type dropout: float, optional
        :param n_targets: number of targets to predict, defaults to 1
        :type n_targets: int, optional
        :param bidirectional: whether to run the GRU bidirectional, defaults to False
        :type bidirectional: bool, optional
        :param post_gru_seq: the layer(s) to apply to the tensor produced by attention weighting, None defaults to a linear transformation
        :type post_gru_seq: Optional[nn.Sequential], optional
        :param batch_first: whether the first dimension in the tensor is batch, defaults to True
        :type batch_first: bool, optional
        """
        super(GRU_with_shared_attention_per_target, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )
        self.att_layers = nn.ModuleList(
            [nn.Linear(hidden_size, 1, bias=True) for _ in range(n_targets)]
        )
        if post_gru_seq is None:
            self.post_gru = nn.Sequential(nn.Linear(n_targets * hidden_size, n_targets))
        else:
            self.post_gru = post_gru_seq

    def forward(self, x):
        x, hn = self.gru(x)
        # x is (batch, seq, hidden)
        x_tmp = []
        for att_lay in self.att_layers:
            logits = att_lay(x)
            att_weights = compute_attention_weights(logits)
            x_tmp.append((x * att_weights).sum(dim=1))
        x = self.post_gru(torch.cat(x_tmp, dim=-1))
        return x


class GRU_with_attention_per_target(nn.Module):
    """A Gated Recurring Unit (GRU) model with attention weighting applied to the last layer to compress all hidden states into a single tensor.
    There are as many attention layers as there are targets and the resulting weighted tensor is kept separate,
    i.e. each attention layer compresses the hidden states of the last layer into a single tensor which is used to predict a single target (and such for each target/attention layer).
    Each weighted hidden state tensor is transformed to predict the target via `post_gru_seq` (which can be any torch Sequential object)."""

    def __init__(
        self,
        input_size: int = 31,
        hidden_size: int = 50,
        num_layers: int = 1,
        dropout: float = 0.0,
        n_targets: int = 2,
        bidirectional: bool = False,
        lin_layers: Optional[nn.Sequential] = None,
        batch_first: bool = True,
        **kwargs
    ):
        """
        :param input_size: size of the input tensor at each step, e.g. number of frequencies, defaults to 31
        :type input_size: int, optional
        :param hidden_size: hidden size to use for GRU, defaults to 50
        :type hidden_size: int, optional
        :param num_layers: number of GRU layers, defaults to 1
        :type num_layers: int, optional
        :param dropout: amount of dropout to apply, defaults to 0.0
        :type dropout: float, optional
        :param n_targets: number of targets to predict, defaults to 2
        :type n_targets: int, optional
        :param bidirectional: whether to run the GRU bidirectional, defaults to False
        :type bidirectional: bool, optional
        :param lin_layers: the last layer to apply to each target-specific attention weighted hidden state tensor (lin_layer is copied for each target), None defaults to a linear transformation for each target
        :type lin_layers: Optional[nn.Sequential], optional
        :param batch_first: whether the first dimension in the tensor is batch, defaults to True
        :type batch_first: bool, optional
        """
        super(GRU_with_attention_per_target, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )
        self.att_layers = nn.ModuleList(
            [nn.Linear(hidden_size, 1, bias=True) for _ in range(n_targets)]
        )
        if lin_layers is None:
            self.lin_layers = nn.ModuleList(
                [nn.Sequential(nn.Linear(hidden_size, 1)) for _ in range(n_targets)]
            )
        else:
            self.lin_layers = nn.ModuleList(
                [copy.deepcopy(lin_layers) for _ in range(n_targets)]
            )

    def forward(self, x):
        x, hn = self.gru(x)
        # x is (batch, seq, hidden)
        x_tmp = []
        for att_lay, lin_layer in zip(self.att_layers, self.lin_layers):
            logits = att_lay(x)
            att_weights = compute_attention_weights(logits)
            x_tmp.append(lin_layer((x * att_weights).sum(dim=1)))
        return torch.cat(x_tmp, dim=-1)
