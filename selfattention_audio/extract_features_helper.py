import torch
import copy
import torch.nn as nn
from selfattention_audio.nn_helpers import SimpleCNN

eps = 1e-7


class ExtractSimpleCNN(nn.Module):
    def __init__(self, nn_arg_list=None):
        super(SimpleCNN, self).__init__()
        if nn_arg_list is None:
            self.layers = nn.Sequential(
                nn.Conv2d(1, 25, (5, 3), padding=(2, 1)),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Linear(25 * 15 * 12, 10),
                nn.Linear(10, 1),
            )
        else:
            self.layers = nn.Sequential(
                *[
                    getattr(nn, nn_name)(*args, **kwargs)
                    for nn_name, args, kwargs in nn_arg_list
                ]
            )

    def forward(self, x):
        return self.layers.forward(x)


class Extract_CNN_GRU_with_multihead_attention_per_target(nn.Module):
    """Class to extract computed attention for each hidden state.
    Should be initialized with the same parameters as the original model
    and its state should be set to the learned weights of the original model.
    See notebook or `fit_model_example.py` for an example."""

    def __init__(
        self,
        cnn=None,
        input_size=100,
        hidden_size=50,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
        n_att=2,
        post_gru_seq=None,
        batch_first=True,
        seq_len=25,
        n_targets=2,
        **kwargs
    ):
        super(Extract_CNN_GRU_with_multihead_attention_per_target, self).__init__()
        if cnn is None:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 100, (3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d((1, 3), (1, 3)),
                nn.Conv2d(100, 100, (3, 5), padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d((1, 5), (1, 5)),
            )
        else:
            self.cnn = cnn
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )
        assert hidden_size % n_att == 0
        self.d_mh = hidden_size // n_att
        self.n_att = n_att
        self.seq_len = seq_len
        self.lin_layer = nn.Linear(hidden_size, hidden_size)
        self.att_layers = nn.ModuleList([nn.Linear(self.d_mh, 1) for _ in range(n_att)])
        if post_gru_seq is None:
            self.post_gru = nn.Sequential(nn.Linear(hidden_size, 10), nn.Linear(10, 1))
        else:
            self.post_gru = post_gru_seq

    def forward(self, x):
        x = self.cnn(x).squeeze()
        x, hn = self.gru(x.permute(0, 2, 1))
        # x is (batch, seq, hidden)
        x_tmp = []
        n_batches = x.size(0)
        x = (
            self.lin_layer(x)
            .view(n_batches, -1, self.n_att, self.d_mh)
            .permute(2, 0, 1, 3)
        )
        # x is now (n_att, batch, seq, d_mh)
        for x_att, att_lay in zip(x, self.att_layers):
            logits = att_lay(x_att)
            ai = torch.exp(logits - torch.max(logits, 1, keepdim=True)[0])
            att_weights = ai / (torch.sum(ai, dim=1, keepdim=True) + eps)
            x_tmp.append(att_weights)
        return torch.stack(x_tmp)


class Extract_CNN_GRU_with_shared_attention_per_target(nn.Module):
    """Class to extract computed attention for each hidden state.
    Should be initialized with the same parameters as the original model
    and its state should be set to the learned weights of the original model.
    See notebook or `fit_model_example.py` for an example."""

    def __init__(
        self,
        cnn=None,
        input_size=100,
        hidden_size=50,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
        post_gru_seq=None,
        batch_first=True,
        n_att=2,
        seq_len=25,
        n_targets=2,
        **kwargs
    ):
        super(Extract_CNN_GRU_with_shared_attention_per_target, self).__init__()
        if cnn is None:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 100, (3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d((1, 3), (1, 3)),
                nn.Conv2d(100, 100, (3, 5), padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d((1, 5), (1, 5)),
            )
        else:
            self.cnn = cnn
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )
        self.att_layers = nn.ModuleList(
            [nn.Linear(hidden_size, 1, bias=True) for _ in range(n_att)]
        )
        if post_gru_seq is None:
            self.post_gru = nn.Sequential(nn.Linear(hidden_size, 10), nn.Linear(10, 1))
        else:
            self.post_gru = post_gru_seq

    def forward(self, x):
        x = self.cnn(x).squeeze()
        x, hn = self.gru(x.permute(0, 2, 1))
        # x is (batch, seq, hidden)
        x_tmp = []
        for att_lay in self.att_layers:
            logits = att_lay(x)
            ai = torch.exp(logits - torch.max(logits, 1, keepdim=True)[0])
            att_weights = ai / (torch.sum(ai, dim=1, keepdim=True) + eps)
            x_tmp.append(att_weights)
        return torch.stack(x_tmp)


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
            self.post_gru = nn.Sequential(nn.Linear(hidden_size, 10), nn.Linear(10, 1))
        else:
            self.post_gru = post_gru_seq

    def forward(self, x):
        x, hn = self.gru(x)
        x_tmp = []
        for att_lay in self.att_layers:
            logits = att_lay(x)
            ai = torch.exp(logits - torch.max(logits, 1, keepdim=True)[0])
            att_weights = ai / (torch.sum(ai, dim=1, keepdim=True) + eps)
            x_tmp.append(att_weights)
        return torch.stack(x_tmp)


class ExtractHiddenGRU_with_shared_attention_per_target(nn.Module):
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
        super(ExtractHiddenGRU_with_shared_attention_per_target, self).__init__()
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
            self.post_gru = nn.Sequential(nn.Linear(2 * hidden_size, n_targets))
        else:
            self.post_gru = post_gru_seq

    def forward(self, x):
        x, hn = self.gru(x)
        x_tmp = []
        for att_lay in self.att_layers:
            logits = att_lay(x)
            ai = torch.exp(logits - torch.max(logits, 1, keepdim=True)[0])
            att_weights = ai / (torch.sum(ai, dim=1, keepdim=True) + eps)
            x_tmp.append((x * att_weights).sum(dim=1))
        return torch.cat(x_tmp, dim=-1)


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
                [
                    nn.Sequential(nn.Linear(hidden_size, 10), nn.Linear(10, 1))
                    for _ in range(n_targets)
                ]
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
            ai = torch.exp(logits - torch.max(logits, 1, keepdim=True)[0])
            att_weights = ai / (torch.sum(ai, dim=1, keepdim=True) + eps)
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
        seq_len=25,
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
        ai = torch.exp(logits - torch.max(logits, 1, keepdim=True)[0])
        att_weights = ai / (torch.sum(ai, dim=1, keepdim=True) + eps)
        return att_weights


class ExtractGRU_attention_linear_encoding_split(nn.Module):
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
        n_patterns=20,
        n_window=5,
        **kwargs
    ):
        super(ExtractGRU_attention_linear_encoding_split, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )
        self.att_layer = nn.Linear(hidden_size, 1, bias=True)
        self.pattern_layer = nn.Sequential(
            nn.Conv2d(1, n_patterns, (n_window, input_size)), nn.ReLU()
        )
        self.n_window = n_window
        if post_gru_seq is None:
            self.post_gru = nn.Sequential(
                #                nn.Linear(n_patterns, 10),
                #                nn.ReLU(),
                nn.Linear(n_patterns, n_targets)
            )
        else:
            self.post_gru = post_gru_seq

    def forward(self, x):
        h, hn = self.gru(torch.squeeze(x))
        # x is (batch, seq, hidden)
        logits = self.att_layer(h[:, self.n_window - 1 :])
        ai = torch.exp(logits - torch.max(logits, 1, keepdim=True)[0])
        att_weights = ai / (torch.sum(ai, dim=1, keepdim=True) + eps)
        # attention weights should be (sequence - n_window + 1)
        x = torch.squeeze(self.pattern_layer(x))
        # x is now (batch, n_patterns, sequence - n_window + 1)
        return (x, att_weights)


class ExtractGRU_attention_max_linear_encoding_split(nn.Module):
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
        n_patterns=20,
        n_window=5,
        **kwargs
    ):
        super(ExtractGRU_attention_max_linear_encoding_split, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )
        self.att_layer = nn.Linear(hidden_size, 1, bias=True)
        self.pattern_layer = nn.Sequential(
            nn.Conv2d(1, n_patterns, (n_window, input_size)), nn.ReLU()
        )
        self.n_window = n_window
        if post_gru_seq is None:
            self.post_gru = nn.Sequential(
                #                nn.Linear(n_patterns, 10),
                #                nn.ReLU(),
                nn.Linear(n_patterns, n_targets)
            )
        else:
            self.post_gru = post_gru_seq

    def forward(self, x):
        h, hn = self.gru(torch.squeeze(x))
        # x is (batch, seq, hidden)
        logits = self.att_layer(h[:, self.n_window - 1 :])
        ai = torch.exp(logits - torch.max(logits, 1, keepdim=True)[0])
        att_weights = ai / (torch.sum(ai, dim=1, keepdim=True) + eps)
        # attention weights should be (sequence - n_window + 1)
        x = torch.squeeze(self.pattern_layer(x))
        # x is now (batch, n_patterns, sequence - n_window + 1)
        return (x, att_weights)
