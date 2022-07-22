import pytest
from typing import Tuple
from numpy.typing import NDArray
import torch


@pytest.fixture
def training_data() -> Tuple[NDArray, NDArray]:
    import joblib

    example_data = joblib.load("./tests/example_data.pkl")
    return example_data["features_train"], example_data["targets_train"]


@pytest.fixture
def single_feature_batch() -> torch.Tensor:
    return torch.rand((10, 25, 30))