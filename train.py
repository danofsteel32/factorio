import copy
import math
import time

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.models as models

import training_images as ti

K = 1
EPOCHS = 5
BATCH_SIZE = 16
MAX_LR = .0005


def get_resnet(num_classes=1):
    # bi-classification only has 1 class. Either is that class or not.
    # Also leaving the entire network trainable not just the last few layers
    model = models.resnet18(pretrained=True)  # pretrained = init with imagenet weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # replace last layer only
    return model


def save_model(model, save_path: Path):
    print(f'Saved model {save_path}')
    torch.save(model.state_dict(), save_path)


def load_model(model, save_path: Path):
    if not save_path.exists():
        raise Exception(f'{save_path} does not exist!')
    model.load_state_dict(torch.load(save_path))
    print(f'Load model {save_path}')
    return model


def train_model():

    # Images / dataloaders
    k_dataloaders = ti.k_dataloaders_iterator(k=K, batch_size=BATCH_SIZE)
    for dataloaders in k_dataloaders:
        # Prep model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_resnet()
        model.to(device)

        # Best loss fn for bi-classification https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        loss_fn = nn.BCEWithLogitsLoss()

        # AdamW explained: https://www.fast.ai/2018/07/02/adam-weight-decay/
        optimizer = optim.AdamW(model.parameters())

        # OneCylceLR papers: https://arxiv.org/pdf/1506.01186.pdf  https://arxiv.org/abs/1708.07120
        steps_per_epoch = len(dataloaders['train'])  # num batches in train set
        scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch)

        # TRAIN
        trained_model, best_loss, train_time = _train(model=model, device=device, epochs=EPOCHS,
                                                      dataloaders=dataloaders, loss_fn=loss_fn,
                                                      optimizer=optimizer, scheduler=scheduler)

        # TEST
        # MCC explained: https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        mcc, test_loss, = _test(device, trained_model, loss_fn, dataloaders)
        save_model(trained_model, Path('model.pth'))


def _train(model, device, epochs, dataloaders, loss_fn, optimizer, scheduler):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.0
    best_mcc = -1.0
    thresh = .5

    # amp = automatic mixed precision (use fp16 instead of fp32 when wont harm accuracy)
    # much faster training
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        print()
        print(f'Epoch {epoch}/{epochs - 1}')
        print('-' * 11)
        # last_epoch = True if epoch == (epochs - 1) else False
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()  # no dropout
            running_loss = 0.0
            tp, tn, fp, fn = 0, 0, 0, 0
            for x, y, _ in dataloaders[phase]:
                x = x.to(device)
                y = y.reshape(-1, 1).to(device)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(phase == 'train'):  # only auto-grad during training
                        outputs = model(x)
                        loss = loss_fn(outputs, y)
                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scale = scaler.get_scale()
                            scaler.update()
                            skip_lr_sched = (scale != scaler.get_scale())
                            if not skip_lr_sched:
                                scheduler.step()

                running_loss += loss.item() * x.size(0)
                epoch_loss = running_loss / len(dataloaders[phase].dataset)

                sg_outputs = torch.sigmoid(outputs.detach())
                for n, o in enumerate(sg_outputs):
                    prob = o.detach()
                    label = int(y.data[n].detach())
                    if prob > thresh and label == 1:  # True Positive
                        tp += 1
                    elif prob <= thresh and label == 0:  # True Negative
                        tn += 1
                    elif prob > thresh and label == 0:  # False Positive
                        fp += 1
                    elif prob <= thresh and label == 1:  # False Negative
                        fn += 1
                    else:
                        print(prob, label)
            try:
                print(f'Conf. Matrix: tp={tp} tn={tn} fp={fp} fn={fn}')
                mcc = (
                        ((tp * tn) - (fp * fn)) /
                        math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
                )
            except ZeroDivisionError:
                print('% by 0 error')
                mcc = best_mcc
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    print('New low loss', best_loss)
                    print('Saved best weights')
                    best_model_wts = copy.deepcopy(model.state_dict())
                if mcc > best_mcc:
                    best_mcc = mcc
            print(f'{phase} Loss: {epoch_loss} MCC: {mcc}')
            print()
    time_elapsed = time.time() - since
    print(f'Trained in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    model.load_state_dict(best_model_wts)
    return model, best_loss, time_elapsed


def _test(device, model, loss_fn, dataloaders) -> tuple[float, float]:
    model.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    thresh = .5
    running_loss = 0.0
    results = []
    for x, y, frames in dataloaders['test']:
        x = x.to(device)
        y = y.reshape(-1, 1).to(device)
        with torch.no_grad():
            outputs = model(x)
            loss = loss_fn(outputs, y)
            running_loss += loss.item() * x.size(0)
            outputs = torch.sigmoid(outputs)
            for n, i in enumerate(outputs):
                prob = float(outputs[n].detach())
                label = int(y.data[n].detach())
                frame = str(frames[n])
                if prob > thresh and label == 1:  # True Positive
                    tp += 1
                elif prob <= thresh and label == 0:  # True Negative
                    tn += 1
                elif prob > thresh and label == 0:  # False Positive
                    fp += 1
                elif prob <= thresh and label == 1:  # False Negative
                    fn += 1
                results.append(f'{frame} {label} {prob}\n')
    try:
        print(f'Conf. Matrix: tp={tp} tn={tn} fp={fp} fn={fn}')
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    except ZeroDivisionError:
        mcc = 0.0
    test_loss = running_loss / len(dataloaders['test'].dataset)
    print('Test MCC ', mcc)
    print('Test Loss ', test_loss)
    with open('test_results.txt', 'w') as f:
        for r in results:
            f.write(r)
    return mcc, test_loss


if __name__ == '__main__':
    train_model()
