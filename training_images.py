from collections import namedtuple
import random
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets import SingleLabelDataset


INPUT_SIZE = 800

TrainingImage = namedtuple('TrainingImage', 'path label')


def get_images():
    pos = [TrainingImage(i, 1) for i in Path('data/neg').iterdir()]
    neg = [TrainingImage(i, 0) for i in Path('data/pos').iterdir()]
    # 50, 30, 20
    random.shuffle(pos)
    random.shuffle(neg)
    train_split = int(len(pos)*.5)
    dev_split = int(len(pos)*.3)
    p_train = pos[:train_split]
    p_dev = pos[train_split:train_split+dev_split]
    p_test = pos[train_split+dev_split:]
    print(f'Positive: train={len(p_train)} dev={len(p_dev)} test={len(p_test)}')

    train_split = int(len(neg)*.5)
    dev_split = int(len(neg)*.3)
    n_train = neg[:train_split]
    n_dev = neg[train_split:train_split+dev_split]
    n_test = neg[train_split+dev_split:]
    print(f'Negative: train={len(n_train)} dev={len(n_dev)} test={len(n_test)}')

    # Oversample minority class to make 50/50 distr. in train set only
    final_p_train = []
    for _ in range(len(n_train)//len(p_train)):
        final_p_train.extend(p_train)
    remainder = len(n_train) % len(p_train)
    final_p_train.extend(random.sample(p_train, remainder))
    print(f'Oversampled Positives: {len(final_p_train) - len(p_train)}')

    train = final_p_train + n_train
    dev = p_dev + n_dev
    test = p_test + n_test
    random.shuffle(train)
    random.shuffle(dev)
    print(f'Total: train={len(train)} dev={len(dev)} test={len(test)}')
    return dict(train=train, dev=dev, test=test)


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
            # transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    }

    datasets = {
        'train': SingleLabelDataset(images['train'], trans['train']),
        'val': SingleLabelDataset(images['dev'], trans['val']),
    }
    dataloaders = {
        x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True,
                      num_workers=5)
        for x in ['train', 'val']
    }
    try:
        test = SingleLabelDataset(images['test'], trans['val'])
        dataloaders['test'] = DataLoader(test, batch_size=batch_size,
                                         num_workers=4)
    except KeyError:
        return dataloaders
    return dataloaders


if __name__ == '__main__':
    images = get_images()
