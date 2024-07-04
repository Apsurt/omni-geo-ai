"""Module handling custom data and loading it into the dataset."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np

if TYPE_CHECKING:
    import torch
import torchvision.io
from torch.utils.data import Dataset
from torchvision import transforms

class CountriesDataset(Dataset):
    """Custom dataset."""

    def __init__(
        self: CountriesDataset,
        train: bool,  # noqa: FBT001
        transform: transforms.Compose | None = None,
        augmenter: transforms.Compose | None = None,
        aug_p: float = 0,
        cache_size: int = 1000,
        ) -> None:
        """Dataset constructor.

        :param self: self
        :type self: CountriesDataset
        :param train: train or validate data
        :type train: bool
        :param transform: transform to apply to data, defaults to None
        :type transform: transforms.Compose | None, optional
        :param augmenter: augments to apply to data, defaults to None
        :type augmenter: transforms.Compose | None, optional
        :param aug_p: probability of applying augment, defaults to 0
        :type aug_p: float, optional
        :param cache_size: number of images to cache in memory, defaults to 1000
        :type cache_size: int, optional
        """
        path = "data/countries/train" if train else "data/countries/validate"

        self.train = train
        self.transform = transform
        self.augmenter = augmenter
        self.aug_p = aug_p
        self.cache_size = cache_size
        self.cache: Dict[int, torch.Tensor] = {}

        dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

        self.data = []
        self.n_classes = len(dirs)
        self.label_dict = {}

        for idx, country in enumerate(dirs):
            self.label_dict[idx] = country
            country_path = os.path.join(path, country)
            files = os.listdir(country_path)
            for file_name in files:
                filepath = os.path.join(country_path, file_name)
                data_tuple = (filepath, idx)
                self.data.append(data_tuple)

        self.n_samples = len(self.data)

    def __getitem__(self: CountriesDataset, index: int) -> Tuple[torch.Tensor, int]:
        """Get item dunder function with caching.

        :param self: self
        :type self: CountriesDataset
        :param index: index to get item from
        :type index: int
        :return: data
        :rtype: Tuple[torch.Tensor, int]
        """
        filepath, label = self.data[index]

        if index in self.cache:
            img = self.cache[index]
        else:
            img = torchvision.io.read_image(filepath)
            if len(self.cache) < self.cache_size:
                self.cache[index] = img

        if self.train and self.augmenter and np.random.rand() <= self.aug_p:
            img = self.augmenter(img)
        if self.transform:
            img = self.transform(img)

        return (img, label)

    def __len__(self: CountriesDataset) -> int:
        """Return number of samples."""
        return self.n_samples