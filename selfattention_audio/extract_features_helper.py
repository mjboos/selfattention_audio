import copy

import torch
import torch.nn as nn

from selfattention_audio.nn_helpers import compute_attention_weights

eps = 1e-7


class ExtractGRU_with_shared_attention_per_target(nn.Module):
    """Class to extract computed attention for each hidden state.
    Should be initialized with the same parameters as the original model
    and its state should be set to the learned weights of the original model.
    See notebook or `fit_model_example.py` for an example."""

    def __init__(
        self,
        input_size=31,
        hidden_size=50,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
        post_gru_seq=None,
        batch_first=True,
        seq_len=25,
        n_targets=2,
        **kwargs
    ):
        super(ExtractGRU_with_shared_attention_per_target, self).__init__()
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
        x_tmp = []
        for att_lay in self.att_layers:
            logits = att_lay(x)
            att_weights = compute_attention_weights(logits)
            x_tmp.append(att_weights)
        return torch.stack(x_tmp)


class ExtractGRU_with_attention_per_target(nn.Module):
    """Class to extract computed attention for each hidden state.
    Should be initialized with the same parameters as the original model
    and its state should be set to the learned weights of the original model.
    See notebook or `fit_model_example.py` for an example."""

    def __init__(
        self,
        input_size=31,
        hidden_size=50,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
        lin_layers=None,
        batch_first=True,
        seq_len=25,
        n_targets=2,
        **kwargs
    ):
        super(ExtractGRU_with_attention_per_target, self).__init__()
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
        x_tmp = []
        for att_lay in self.att_layers:
            logits = att_lay(x)
            att_weights = compute_attention_weights(logits)
            x_tmp.append(att_weights)
        return torch.stack(x_tmp)


class ExtractGRU_with_attention(nn.Module):
    """Class to extract computed attention for each hidden state.
    Should be initialized with the same parameters as the original model
    and its state should be set to the learned weights of the original model.
    See notebook or `fit_model_example.py` for an example."""

    def __init__(
        self,
        input_size=31,
        hidden_size=50,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
        post_gru_seq=None,
        batch_first=True,
        n_targets=2,
        **kwargs
    ):
        super(ExtractGRU_with_attention, self).__init__()
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
        return att_weights
