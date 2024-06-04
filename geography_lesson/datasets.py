import os
import numpy as np
import torch
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
    def __init__(self, train: bool) -> None:
        if train:
            path = "data/countries/train"
        else:
            path = "data/countries/validate"
            
        dirs = os.listdir(path)
        for _dir in dirs:
            if not os.path.isdir(os.path.join(path, _dir)):
                dirs.remove(_dir)
        
        self.inputs = []
        self.labels = []
        self.n_classes = len(dirs)
        self.label_dict = {}
        
        for idx, country in enumerate(dirs):
            print(country)
            self.label_dict[idx] = country
            country_label = torch.tensor(idx, dtype=torch.long)
            files = os.listdir(os.path.join(path, country))
            for _file in files:
                filepath = os.path.join(path,country,_file)
                pil_img = Image.open(filepath)
                np_img = np.array(pil_img, dtype=np.float32)/255
                pil_img.close()
                r,g,b = np_img[:, :, 0], np_img[:, :, 1], np_img[:, :, 2]
                np_img = np.array(([r,g,b]))
                img = torch.from_numpy(np_img)
                self.inputs.append(img)
                self.labels.append(country_label)
        self.n_samples = len(self.inputs)

    def __getitem__(self, index) -> torch.tensor:
        return self.inputs[index], self.labels[index]

    def __len__(self) -> int:
        return self.n_samples

if __name__ == "__main__":
    training_dataset = CountriesDataset(train=True)
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=16, shuffle=True)