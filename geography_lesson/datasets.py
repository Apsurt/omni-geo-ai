"""Module handling custom data and loading it into the dataset."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, Tuple
import time
import random
from tqdm import tqdm

import numpy as np

if TYPE_CHECKING:
    import torch
import torchvision.io
from torch.utils.data import Dataset
from torchvision import transforms

from device import get_device

class CountriesDataset(Dataset):
    """Custom dataset."""

    def __init__(
        self: CountriesDataset,
        train: bool,  # noqa: FBT001
        transform: transforms.Compose | None = None,
        augmenter = None,
        aug_p: float = 0,
        cache_size: int = 1000,
        debug: bool = False,
        preload: bool = False,
        ) -> None:
        """Dataset constructor.

        :param self: self
        :type self: CountriesDataset
        :param train: train or validate data
        :type train: bool
        :param transform: transform to apply to data, defaults to None
        :type transform: transforms.Compose | None, optional
        :param augmenter: augments to apply to data, defaults to None
        :type augmenter: optional
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
        self.debug = debug
        self.cache_hits = 0
        self.total_accesses = 0
        self.device = get_device()

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

        if self.cache_size > self.n_samples:
            self.cache_size = self.n_samples

        if self.debug:
            print(f"Dataset initialized with {self.n_samples} samples")
            print(f"Cache size: {self.cache_size}")
        
        if preload:
            self.preload_cache()

    def preload_cache(self):
        """Preload images into cache."""
        print("Preloading cache...")
        start_time = time.time()
        indices_to_cache = random.sample(range(self.n_samples), min(self.cache_size, self.n_samples))
        
        for idx in tqdm(indices_to_cache, desc="Preloading cache", disable=not self.debug):
            filepath, _ = self.data[idx]
            img = torchvision.io.read_image(filepath)
            img.to(self.device)
            self.cache[idx] = img
        
        end_time = time.time()
        print(f"Cache preloading completed. Time taken: {end_time - start_time:.2f} seconds")
        print(f"Cache size after preloading: {len(self.cache)}")

    def __getitem__(self: CountriesDataset, index: int) -> Tuple[torch.Tensor, int]:
        """Get item dunder function with caching.

        :param self: self
        :type self: CountriesDataset
        :param index: index to get item from
        :type index: int
        :return: data
        :rtype: Tuple[torch.Tensor, int]
        """
        self.total_accesses += 1
        filepath, label = self.data[index]

        start_time = time.time()

        if index in self.cache.keys():
            img = self.cache[index]
            self.cache_hits += 1
            #if self.debug:
            #    print(f"Cache hit for index {index}")
        else:
            img = torchvision.io.read_image(filepath)
            if len(self.cache) < self.cache_size:
                self.cache[index] = img
            #    if self.debug:
            #        print(f"Added image {index} to cache. Cache size: {len(self.cache)}")
            #elif self.debug:
            #    print(f"Cache full, couldn't add image {index}")

        load_time = time.time() - start_time

        if self.train and self.augmenter and np.random.rand() <= self.aug_p:
            aug_start = time.time()
            img = self.augmenter(img)
            aug_time = time.time() - aug_start
        else:
            aug_time = 0
        if self.transform:
            transform_start = time.time()
            img = self.transform(img)
            transform_time = time.time() - transform_start
        else:
            transform_time = 0
        
        #if self.debug:
        #    print(f"Index: {index}, Load time: {load_time:.4f}s, Aug time: {aug_time:.4f}s, Transform time: {transform_time:.4f}s")
        #    print(f"Cache hit rate: {self.cache_hits / self.total_accesses:.2%}")

        return (img, label)

    def __len__(self: CountriesDataset) -> int:
        """Return number of samples."""
        return self.n_samples
    
    def get_cache_stats(self):
        """Return cache statistics."""
        return {
            "cache_size": len(self.cache),
            "cache_limit": self.cache_size,
            "cache_hits": self.cache_hits,
            "total_accesses": self.total_accesses,
            "hit_rate": self.cache_hits / self.total_accesses if self.total_accesses > 0 else 0
        }