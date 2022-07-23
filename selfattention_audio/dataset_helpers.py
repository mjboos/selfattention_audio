from typing import Callable, Optional, Tuple

import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

TransformCallable = Callable[[NDArray, NDArray], Tuple[NDArray, NDArray]]


class AudioBrainDataset(Dataset):
    """A dataset containing (lagged) audio features and neural activity of some kind."""

    def __init__(
        self,
        audio_features: NDArray,
        brain_activity: NDArray,
        transform: Optional[TransformCallable] = None,
    ):
        """A dataset class for deep encoding models.

        :param audio_features: array of shape (samples, time_steps, frequencies)
        :type audio_features: NDArray
        :param brain_activity: array of shape (samples, brain_dimensions)
        :type brain_activity: NDArray
        :param transform: _description_, defaults to None
        :type transform: Optional[TransformCallable], optional
        """
        self.features = audio_features
        self.targets = brain_activity
        self.transform = transform

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.transform:
            sample = (self.transform(self.features[idx]), self.targets[idx])
        else:
            sample = (self.features[idx], self.targets[idx])
        return sample
