from PIL import Image
import torch
from torch.utils.data import Dataset


class SingleLabelDataset(Dataset):

    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        X = self.transform(Image.open(image.path))
        Y = self.images[index].label
        return X, torch.tensor(Y, dtype=torch.float32), image.path
