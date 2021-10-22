from collections import namedtuple
import random
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets import SingleLabelDataset


INPUT_SIZE = 800

TrainingImage = namedtuple('TrainingImage', 'path label')


def k_dataloaders_iterator(k: int = 1, batch_size: int = 16):
    # yields k random sampled (train, dev) dataloaders with constant test set
    pos = [TrainingImage(i, 1) for i in Path('data/neg').iterdir()]
    neg = [TrainingImage(i, 0) for i in Path('data/pos').iterdir()]
    # Hold 20% for test set
    p_test_split = int(len(pos)*.2)
    p_test = pos[:p_test_split]
    p_rest = pos[p_test_split:]

    n_test_split = int(len(neg)*.2)
    n_test = neg[:n_test_split]
    n_rest = neg[n_test_split:]

    test = p_test + n_test

    folds = k_fold(k, p_rest+n_rest)
    for fold in folds:
        fold['test'] = test
        yield get_dataloaders(fold, batch_size)


def k_fold(k: int, images: list[TrainingImage]):
    # split images into k train and dev sets. test set is held constant

    random.shuffle(images)
    # 80, 20
    train_sample = int(len(images)*.8)

    out = []
    # print(len(images))
    for n in range(k):
        train_ = set(random.sample(images, train_sample))
        dev_ = set(images) - train_

        train, dev = list(train_), list(dev_)
        # print(f'Train negs: {len(train)}')
        p_train = [i for i in train if i.label == 1]
        # print(f'Train pos: {len(p_train)}')

        # Oversample pos to match neg in train set
        for _ in range((len(train)) // len(p_train)):
            train.extend(p_train)
        remainder = len(train) % len(p_train)
        train.extend(random.sample(p_train, remainder))

        # print(f'Train total {len(train)}')
        random.shuffle(train)
        out.append({'train': train, 'dev': dev})
    return out


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
        'dev': transforms.Compose([
            transforms.CenterCrop((INPUT_SIZE, INPUT_SIZE)),
            # transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    }

    datasets = {
        'train': SingleLabelDataset(images['train'], trans['train']),
        'val': SingleLabelDataset(images['dev'], trans['dev']),
    }
    dataloaders = {
        x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True,
                      num_workers=4)
        for x in ['train', 'val']
    }
    try:
        test = SingleLabelDataset(images['test'], trans['dev'])
        dataloaders['test'] = DataLoader(test, batch_size=batch_size,
                                         num_workers=4)
    except KeyError:
        return dataloaders
    return dataloaders


if __name__ == '__main__':
    for i in k_dataloaders_iterator(k=1):
        i['dev']
