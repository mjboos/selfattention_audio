import numpy as np
import torch
from selfattention_audio import dataset_helpers as dhelp
from selfattention_audio import nn_helpers as nnhelp
from selfattention_audio import lightning_modules as light
from pytorch_lightning import Trainer
from selfattention_audio import extract_features_helper as ext
import matplotlib.pyplot as plt
from torch.nn import Module
from typing import Mapping, Optional
from numpy.typing import NDArray
from matplotlib.figure import Figure

# TODO: make installable
# TODO: write tests
# TODO: add other engineering garnishes
# TODO: copy some model architecture plot from presentation


# hard coded mel frequencies (for example data) so we don't need another dependency
mel_freqs = np.array(
    [
        0,
        101,
        202,
        302,
        403,
        503,
        604,
        704,
        805,
        905,
        1006,
        1116,
        1238,
        1373,
        1523,
        1689,
        1874,
        2079,
        2306,
        2558,
        2837,
        3147,
        3491,
        3872,
        4295,
        4764,
        5284,
        5862,
        6502,
        7213,
        8000,
    ]
)


def plot_attention_with_specgram(
    attention: NDArray,
    features: NDArray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    att_max: Optional[float] = None,
) -> Figure:
    """Plots attention weights and the corresponding feature spectrogram.

    :param attention: Attention weights of shape (n_timesteps, )
    :type attention: NDArray
    :param features: Spectrogram features of shape (n_timesteps, n_frequencies)
    :type features: NDArray
    :param vmin: min value for spectrogram image plot, defaults to None
    :type vmin: Optional[float], optional
    :param vmax: max value for spectrogram image plot, defaults to None
    :type vmax: Optional[float], optional
    :param att_max: max attention value to include in plot, defaults to None
    :type att_max: Optional[float], optional
    :return: the figure object of the attention plot
    :rtype: Figure
    """

    fig = plt.figure(constrained_layout=False, figsize=(10, 7))
    gs1 = fig.add_gridspec(nrows=4, ncols=3)
    spec_ax = fig.add_subplot(gs1[:2, :])
    att_ax = fig.add_subplot(gs1[2:, :])
    spec_ax.imshow(features.T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    spec_ax.set_xticks([0, 10, 20])
    spec_ax.set_xticklabels(["-500 ms", "-300 ms", "-100 ms"])
    spec_ax.set_ylabel("Frequencies in Hz")
    spec_ax.set_yticks([1, 5, 10, 15, 20, 25, 30])
    spec_ax.set_yticklabels(
        ["{}".format(fr) for fr in mel_freqs[[1, 5, 10, 15, 20, 25, 30]]]
    )
    att_ax.plot(attention, c="k")
    att_ax.set_xticks([0, 10, 20])
    att_ax.set_xticklabels(["-500 ms", "-300 ms", "-100 ms"])
    if att_max is not None:
        att_ax.set_ylim(0, att_max)

    fig.tight_layout()
    return fig


def create_lightning_model(
    trainset: dhelp.AudioBrainDataset,
    testset: dhelp.AudioBrainDataset,
    valset: dhelp.AudioBrainDataset,
    nn_model: Optional[Module] = None,
    train_loader_args: Optional[Mapping] = None,
    test_loader_args: Optional[Mapping] = None,
    val_loader_args: Optional[Mapping] = None,
    lr: float = 3e-3,
    **kwargs,
) -> light.AuditoryEncodingLightning:
    """Creates an FGLightning object that contains the model and train, test, and val datasets and loaders for training & testing.

    :param trainset: dataset with training data
    :type trainset: dhelp.AudioBrainDataset
    :param testset: dataset with test data
    :type testset: dhelp.AudioBrainDataset
    :param valset: dataset with validation data
    :type valset: dhelp.AudioBrainDataset
    :param nn_model: the neural network model to use, None defaults to a GRU with attention
    :type nn_model: Optional[Module], optional
    :param train_loader_args: arguments for the train loader, defaults to None
    :type train_loader_args: Optional[Mapping], optional
    :param test_loader_args: arguments for the test loader, defaults to None
    :type test_loader_args: Optional[Mapping], optional
    :param val_loader_args: arguments for the validation loader, defaults to None
    :type val_loader_args: Optional[Mapping], optional
    :param lr: learning rate to use for the Adam optimizer, defaults to 3e-3
    :type lr: float, optional
    :return: lightning model for running auditory encoding
    :rtype: light.AuditoryEncodingLightning
    """
    loader_default_args = dict(batch_size=512, shuffle=True, num_workers=2)
    train_loader_args = (
        loader_default_args if train_loader_args is None else train_loader_args
    )
    test_loader_args = (
        loader_default_args if test_loader_args is None else test_loader_args
    )
    val_loader_args = (
        loader_default_args if val_loader_args is None else val_loader_args
    )

    # use default model found by hypterparameter search
    if nn_model is None:
        nn_model = nnhelp.GRU_with_attention_per_target(
            hidden_size=150,
            num_layers=2,
            n_patterns=30,
            n_window=5,
            n_targets=2,
            dropout=0.2,
        )

    model = light.AuditoryEncodingLightning(
        trainset,
        testset,
        valset,
        model=nn_model,
        loss_func=None,
        train_loader_args=train_loader_args,
        test_loader_args=test_loader_args,
        val_loader_args=val_loader_args,
        lr=lr,
    )
    return model


# fits example model with the same parameters that were used in later interpretability analysis
if __name__ == "__main__":
    from pytorch_lightning.callbacks import ModelCheckpoint
    import joblib

    example_data = joblib.load("./tests/example_data.pkl")
    train_data, test_data, val_data = [
        dhelp.AudioBrainDataset(
            example_data[f"features_{split}"],
            example_data[f"targets_{split}"],
            transform=None,
        )
        for split in ["train", "test", "val"]
    ]
    model_params = dict(
        hidden_size=150,
        num_layers=2,
        n_patterns=30,
        n_window=5,
        n_targets=2,
        dropout=0.2,
    )

    nn_model = nnhelp.GRU_with_attention(**model_params)
    model = create_lightning_model(train_data, test_data, val_data, nn_model=nn_model)

    checkpoint_callback = ModelCheckpoint(
        filepath="./saved_models/example_model_gru_single_head_attention",
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="gru_single_head_attention",
    )

    trainer = Trainer(max_nb_epochs=100, checkpoint_callback=checkpoint_callback)
    trainer.fit(model)

    gru_att_extractor = create_lightning_model(
        train_data,
        test_data,
        val_data,
        nn_model=ext.ExtractGRU_with_attention(**model_params),
    )
    gru_att_extractor.load_state_dict(model.state_dict())
    # extract attention weights for each sample in the test set
    attention = np.squeeze(
        gru_att_extractor.model(torch.tensor(test_data.features)).detach().numpy()
    )

    sample_to_plot = 200
    _ = plot_attention_with_specgram(
        attention[sample_to_plot],
        test_data.features[sample_to_plot],
    )
