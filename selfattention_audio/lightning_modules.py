from typing import Callable, Mapping, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from torch.tensor import Tensor

import selfattention_audio.dataset_helpers as dhelp
import selfattention_audio.nn_helpers as nnhelp

LossFunc = Callable[[Tensor, Tensor], Tensor]


class AuditoryEncodingLightning(pl.LightningModule):
    """An auditory encoding lightning model that simplifies training and validation."""

    def __init__(
        self,
        trainset: dhelp.AudioBrainDataset,
        testset: dhelp.AudioBrainDataset,
        valset: dhelp.AudioBrainDataset,
        model: Optional[Module] = None,
        train_loader_args: Optional[Mapping] = None,
        test_loader_args: Optional[Mapping] = None,
        val_loader_args: Optional[Mapping] = None,
        loss_func: Optional[LossFunc] = None,
        lr: float = 3e-5,
        **kwargs,
    ):
        """
        :param trainset: dataset with training data
        :type trainset: dhelp.AudioBrainDataset
        :param testset: dataset with test data
        :type testset: dhelp.AudioBrainDataset
        :param valset: dataset with validation data
        :type valset: dhelp.AudioBrainDataset
        :param model: the neural network model to use, None defaults to a GRU with attention
        :type model: Optional[Module], optional
        :param train_loader_args: arguments for the train loader, defaults to None
        :type train_loader_args: Optional[Mapping], optional
        :param test_loader_args: arguments for the test loader, defaults to None
        :type test_loader_args: Optional[Mapping], optional
        :param val_loader_args: arguments for the validation loader, defaults to None
        :type val_loader_args: Optional[Mapping], optional
        :param loss_func: the loss function to use, None defaults to torch's MSELoss
        :type loss_func: Optional[LossFunc], optional
        :param lr: learning rate to use for the Adam optimizer, defaults to 3e-3
        :type lr: float, optional
        """
        super(AuditoryEncodingLightning, self).__init__(**kwargs)
        self.model = nnhelp.SimpleGRU() if model is None else model
        self.loss_func = nn.MSELoss() if loss_func is None else loss_func
        self.train_loader_args = {} if train_loader_args is None else train_loader_args
        self.test_loader_args = {} if test_loader_args is None else test_loader_args
        self.val_loader_args = {} if val_loader_args is None else val_loader_args
        self.trainset = trainset
        self.lr = lr
        self.testset = testset
        self.valset = valset

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.model(x)
        return {"val_loss": self.loss_func(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        print(f"Validation loss: {avg_loss}")
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self, **kwargs):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=self.lr, **kwargs)

    @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, **self.train_loader_args)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return torch.utils.data.DataLoader(self.valset, **self.val_loader_args)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return torch.utils.data.DataLoader(self.testset, **self.test_loader_args)
