import os
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageNetDataset(Dataset):
    def __init__(self) -> None:
        pass

    def __getitem__(self, index) -> torch.tensor:
        pass

    def __len__(self) -> int:
        pass

class CountriesDataset(Dataset):
    def __init__(self, device) -> None:
        path = "data/temp"
        dirs = os.listdir(path)
        for _dir in dirs:
            if not os.path.isdir(os.path.join(path, _dir)):
                dirs.remove(_dir)
        
        self.inputs = []
        self.labels = []
        self.n_classes = len(dirs)
        self.label_dict = {}
        
        for idx, country in enumerate(dirs):
            #print(country)
            self.label_dict[idx] = country
            country_label = [0]*self.n_classes
            country_label[idx] = 1
            country_label = torch.tensor(country_label, dtype=torch.float32)
            country_label.to(device)
            files = os.listdir(os.path.join(path, country))
            for _file in files:
                filepath = os.path.join(path,country,_file)
                np_img = np.array(Image.open(filepath), dtype=np.float32)/255
                np_img.resize((1024, 2048, 3))
                img = torch.from_numpy(np_img)
                img.to(device)
                self.inputs.append(img)
                self.labels.append(country_label)
        self.n_samples = len(self.inputs)

    def __getitem__(self, index) -> torch.tensor:
        return self.inputs[index], self.labels[index]

    def __len__(self) -> int:
        return self.n_samples

if __name__ == "__main__":
    device = torch.device("mps")
    training_dataset = CountriesDataset(device)
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=16, shuffle=True)