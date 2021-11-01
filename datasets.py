from collections import namedtuple
from pathlib import Path
from PIL import Image
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


INPUT_SIZE = 800  # Center cropping images bc center is all we care about in this case
STOP = 2220  # point in video where press stops (no need to include 200 of same thing)

# Just make life easier
TrainingImage = namedtuple('TrainingImage', 'path label')


def ordered_frames():
    frames = [frame for frame in Path('data/frames').iterdir()]
    return sorted(frames, key=lambda f: int(f.name.split('.')[0]))[:STOP]


def get_crashes() -> tuple[list, list]:
    # returns test crashes and rest of crashes separately
    with open('crash_frames.txt', 'r') as f:
        lines = f.readlines()

    test, rest = [], []
    for c in lines:
        path = Path(c.strip('\n'))
        if int(path.name.split('.')[0].split('/')[-1]) < 2037:
            test.append(TrainingImage(path, 1))
        else:
            rest.append(TrainingImage(path, 1))
    return test, rest


def get_oks() -> list[TrainingImage]:
    with open('crash_frames.txt', 'r') as f:
        crashes = set(Path(c.strip('\n')) for c in f.readlines())
    return [TrainingImage(p, 0) for p in set(ordered_frames()) - crashes]


def splits(oversample: bool = False) -> dict:
    # 60/20/20
    # TODO: support undersampling oks as well
    oks = get_oks()
    crashes = get_crashes()

    ok_split = int(len(oks) * .2)
    crash_split = int(len(crashes[1]) * .2)

    test = crashes[0] + oks[:ok_split]
    val = crashes[1][:crash_split] + oks[ok_split:ok_split*2]
    train = crashes[1][crash_split:] + oks[ok_split*2:]

    if oversample:
        # Make copies of crashes in training set only to reach 50/50 dist.
        # With aggressive augmentation this doesn't have to be an issue
        crash_train = [c for c in train if c.label == 1]
        for _ in range((len(train) - len(crash_train)) // len(crash_train)):
            train.extend(crash_train)

    return dict(train=train, val=val, test=test)


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
        frame_id = int(image.path.name.split('.')[0])
        return X, torch.tensor(Y, dtype=torch.float32), frame_id


def all_frames_dataloader(batch_size):
    images = [TrainingImage(p, 0) for p in ordered_frames()]
    transforms_ = transforms.Compose([
        transforms.CenterCrop((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test = SingleLabelDataset(images, transforms_)
    return DataLoader(test, batch_size=batch_size, num_workers=8, pin_memory=True)


def get_dataloaders(images, batch_size):
    trans = {
        'train': transforms.Compose([
            transforms.RandomRotation(degrees=50, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomGrayscale(p=.1),
            transforms.RandomInvert(p=.3),
            transforms.RandomHorizontalFlip(p=.4),
            transforms.RandomVerticalFlip(p=.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    }

    datasets = {
        'train': SingleLabelDataset(images['train'], trans['train']),
        'val': SingleLabelDataset(images['val'], trans['val']),
    }
    dataloaders = {
        x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True,
                      num_workers=12, pin_memory=True)
        for x in ['train', 'val']
    }
    try:
        test = SingleLabelDataset(images['test'], trans['val'])
        dataloaders['test'] = DataLoader(test, batch_size=batch_size,
                                         num_workers=8, pin_memory=True)
    except KeyError:
        return dataloaders
    return dataloaders


if __name__ == '__main__':
    splits_ = splits(True)
    for s in splits_:
        print(s, len(splits_[s]), sum([i.label for i in splits_[s]]))
