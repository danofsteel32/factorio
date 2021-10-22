from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from matplotlib import pyplot
import skimage.transform

from train import get_resnet, load_model

INPUT_SIZE = 800


class SaveFeatures():
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()

    def remove(self):
        self.hook.remove()


def getCAM(feature_conv, weight_fc):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc.dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]


def single_image(image_path: Path):
    trans = transforms.Compose([
            transforms.CenterCrop((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    just_resize = transforms.Compose([transforms.CenterCrop((INPUT_SIZE, INPUT_SIZE))])
    img = Image.open(image_path)
    return just_resize(img), trans(img)


if __name__ == '__main__':
    image_path = Path('data/neg/90.jpg')
    original_image, processed_image = single_image(image_path)
    model = get_resnet()
    try:
        trained_model = load_model(model, Path('model.pth'))
    except:
        model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
        trained_model = model
    final_layer = trained_model._modules.get('layer4')
    activated_features = SaveFeatures(final_layer)
    prediction_var = Variable((processed_image.unsqueeze(0)), requires_grad=True)
    pred = model(prediction_var)
    pred = torch.sigmoid(pred).data.squeeze()
    activated_features.remove()
    weight_softmax_params = list(model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    overlay = getCAM(activated_features.features, weight_softmax)

    # pyplot.imshow(overlay[0], alpha=0.5, cmap='jet')
    pyplot.imshow(original_image)
    pyplot.imshow(skimage.transform.resize(overlay[0], processed_image.shape[1:3]), alpha=0.5, cmap='jet');
    pyplot.savefig('heatmap.png')
