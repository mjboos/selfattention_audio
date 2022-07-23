import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from selfattention_audio import dataset_helpers as dhelp
from selfattention_audio import lightning_modules as light
from selfattention_audio import nn_helpers as nhelp

models_to_test = [
    "GRU_with_attention",
]


@pytest.mark.parametrize("model_class", models_to_test)
def test_training_procedure(model_class, full_data, tmp_saved_models_path) -> None:
    datasets = [dhelp.AudioBrainDataset(*data) for data in full_data]

    model = getattr(nhelp, model_class)(input_size=datasets[0][0][0].shape[-1])
    light_model = light.AuditoryEncodingLightning(
        datasets[0], datasets[1], datasets[2], model=model
    )
    initial_params = {
        name: p.clone()
        for (name, p) in light_model.model.named_parameters()
        if p.requires_grad
    }

    checkpoint_callback = ModelCheckpoint(
        filepath=f"{tmp_saved_models_path}/{model_class}",
        prefix=model_class,
    )

    trainer = Trainer(max_epochs=1, checkpoint_callback=checkpoint_callback)
    trainer.fit(light_model)

    # now check for nans and inf - since a training step has already run
    param_dict = {
        name: p for (name, p) in light_model.model.named_parameters() if p.requires_grad
    }
    for name, p in param_dict.items():
        assert not torch.any(torch.isinf(p))
        assert not torch.any(torch.isnan(p))
        if p.requires_grad:
            assert not torch.equal(p, initial_params[name])
