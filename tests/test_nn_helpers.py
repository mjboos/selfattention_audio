import pytest
import torch

from selfattention_audio import nn_helpers as nhelp

models_to_test = [
    "GRU_with_attention",
    "GRU_with_shared_attention_per_target",
    "GRU_with_attention_per_target",
    "SimpleGRU",
    "GRU_with_concat_attention_per_target",
]


def run_training_step(model, optim, loss_fn, inputs, labels):
    optim.zero_grad()
    outputs = model(inputs)

    loss = loss_fn(outputs, labels)
    loss.backward()

    # Adjust learning weights
    optim.step()


def assert_parameters_update(model, optim, loss_fn, inputs, labels):
    # get initial params
    initial_params = {
        name: p.clone() for (name, p) in model.named_parameters() if p.requires_grad
    }
    # perform training step
    run_training_step(model, optim, loss_fn, inputs, labels)
    for name, updated_p in model.named_parameters():
        if updated_p.requires_grad:
            assert not torch.equal(updated_p, initial_params[name])


def test_compute_attention_weights() -> None:
    x = torch.rand((1, 15, 30))
    x = x / x.sum(dim=1, keepdim=True)

    learned_weights = x[:, 10, :].squeeze()
    logits = (x * learned_weights).sum(dim=-1)
    weights = nhelp.compute_attention_weights(logits)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert weights.min() >= 0 and weights.max() <= 1


@pytest.mark.parametrize("model_class", models_to_test)
def test_model(model_class, training_data) -> None:
    training_data = [torch.tensor(data) for data in training_data]
    model = getattr(nhelp, model_class)(input_size=training_data[0].shape[-1])
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())
    assert_parameters_update(model, optim, loss_fn, training_data[0], training_data[1])

    # now check for nans and inf - since a training step has already run
    param_dict = {name: p for (name, p) in model.named_parameters() if p.requires_grad}
    for name, p in param_dict.items():
        assert not torch.any(torch.isinf(p))
        assert not torch.any(torch.isnan(p))
