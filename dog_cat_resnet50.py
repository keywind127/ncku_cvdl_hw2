import torch 

from typing import *

import numpy as np

from torchvision import transforms

from torchsummary import summary

from resnet50_utils import *

import os

def load_dog_cat_model(model_name : str) -> torch.nn.Module:

    assert isinstance(model_name, str)

    model = torch.load(model_name, map_location = torch.device("cpu"))

    return model

def predict_dog_cat_image(model : torch.nn.Module, image : np.ndarray) -> Tuple[ int, np.ndarray ]:

    assert isinstance(model, torch.nn.Module)

    assert isinstance(image, np.ndarray)

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels = 3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
    ])

    image_test = transform_test(image).unsqueeze(0)

    model.eval()

    outputs = model(image_test).detach().numpy()[0]

    best_prediction = 1 * (outputs[0] >= 0.5)

    return (best_prediction, outputs)

if (__name__ == "__main__"):

    model_name = os.path.join(os.path.dirname(__file__), "models/resnet50_bn_dogs_cats_d172023_2.pt")

    model = load_dog_cat_model(model_name)

    summary(model, (3, 224, 224))