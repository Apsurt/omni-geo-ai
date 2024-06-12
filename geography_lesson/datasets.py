import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.io

class CountriesDataset(Dataset):
    def __init__(self, train: bool, transform=None, augmenter=None, aug_p=0) -> None:
        if train:
            path = "data/countries/train"
        else:
            path = "data/countries/validate"
        
        self.transform = transform
        self.augmenter = augmenter
        self.aug_p = aug_p
        
        dirs = os.listdir(path)
        for _dir in dirs:
            if not os.path.isdir(os.path.join(path, _dir)):
                dirs.remove(_dir)
        
        self.data = []
        self.n_classes = len(dirs)
        self.label_dict = {}

        for idx, country in enumerate(dirs):
            self.label_dict[idx] = country
            country_path = os.path.join(path, country)
            files = os.listdir(country_path)
            for file_name in files:
                filepath = os.path.join(country_path, file_name)
                data_tuple = (filepath,idx)
                self.data.append(data_tuple)
        
        self.n_samples = len(self.data)

    def __getitem__(self, index) -> torch.tensor:
        filepath, label = self.data[index]
        img = torchvision.io.read_image(filepath)
        if self.augmenter:
            if np.random.rand() <= self.aug_p:
                img = self.augmenter(img)
        if self.transform:
            img = self.transform(img)
        
        return (img, label)

    def __len__(self) -> int:
        return self.n_samples

if __name__ == "__main__":
    from torchvision import transforms
    transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.ToPILImage()
    ])

    augmenter = transforms.AugMix()

    training_set = CountriesDataset(train=True, transform=transform, augmenter=augmenter)
    training_dataloader = DataLoader(dataset=training_set, batch_size=8, shuffle=True)
    for data in training_dataloader:
        inputs, label = data
        exit()