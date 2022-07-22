from selfattention_audio import nn_helpers as nhelp
from selfattention_audio import extract_features_helper as extr
import torch
import pytest

models_to_test = [
    "GRU_with_attention",
    "GRU_with_shared_attention_per_target",
    "GRU_with_attention_per_target",
]


@pytest.mark.parametrize("original_class", models_to_test)
def test_extraction(original_class, single_feature_batch):
    nn_class = getattr(nhelp, original_class)
    # test that each class has an extractor class
    extractor = getattr(extr, "Extract" + original_class)
    nn_instance = nn_class(input_size=30)
    extractor_instance = extractor(input_size=30)
    extractor_instance.load_state_dict(nn_instance.state_dict())
    attention = extractor_instance(single_feature_batch).squeeze()
    assert attention.max() <= 1 and attention.min() <= 1
    if len(attention.shape) == 2:
        assert attention.shape[0] == single_feature_batch.shape[0]
        assert attention.shape[1] == single_feature_batch.shape[1]
    else:
        assert attention.shape[1] == single_feature_batch.shape[0]
        assert attention.shape[2] == single_feature_batch.shape[1]
