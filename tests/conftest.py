from typing import List, Tuple

import pytest
import torch
from numpy.typing import NDArray


@pytest.fixture(scope="session")
def tmp_saved_models_path(tmp_path_factory):
    model_path = tmp_path_factory.mktemp("saved_models")
    return model_path


@pytest.fixture
def full_data() -> List[Tuple[NDArray, NDArray]]:
    import joblib

    example_data = joblib.load("./tests/example_data.pkl")
    return [
        (example_data[f"features_{tp}"], example_data[f"targets_{tp}"])
        for tp in ["train", "val", "test"]
    ]


@pytest.fixture
def training_data() -> Tuple[NDArray, NDArray]:
    import joblib

    example_data = joblib.load("./tests/example_data.pkl")
    return example_data["features_train"], example_data["targets_train"]


@pytest.fixture
def single_feature_batch() -> torch.Tensor:
    return torch.rand((10, 25, 30))
