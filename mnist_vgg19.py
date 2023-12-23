from torchsummary import summary

from torchvision import transforms

from matplotlib import pyplot as plt

from vgg19_utils import *

from PIL import Image

from typing import *

import numpy as np

import torch

import cv2

import os

def load_mnist_model(model_name : str) -> torch.nn.Module:

    assert isinstance(model_name, str)

    model = torch.load(model_name, map_location = torch.device("cpu"))

    return model

def predict_mnist_image(model : torch.nn.Module, image : np.ndarray) -> Tuple[ int, np.ndarray ]:

    assert isinstance(model, torch.nn.Module)

    assert isinstance(image, np.ndarray)

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels = 3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image_test = transform_test(image).unsqueeze(0)

    model.eval()

    outputs = model(image_test).detach().numpy()[0]

    best_prediction = np.argmax(outputs)

    return (best_prediction, outputs)

mnist_classes = [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" ]

def draw_mnist_distribution(probabilities : List[ float ]) -> np.ndarray:

    global mnist_classes

    plt.bar(mnist_classes, probabilities, alpha = 0.7)

    plt.ylim(0, 1)

    plt.title("Probability Distribution for MNIST-10 Classes")

    plt.xlabel("Class")

    plt.ylabel("Probability")

    plt.xticks(rotation = 30)

    figure = plt.gcf()

    figure.canvas.draw()

    image_plot = cv2.cvtColor(np.uint8(figure.canvas.renderer._renderer), cv2.COLOR_RGB2BGR)

    plt.clf()

    return image_plot

if (__name__ == "__main__"):

    model_name = os.path.join(os.path.dirname(__file__), "models/vgg19_bn_MNIST_d162023.pt")

    model = load_mnist_model(model_name)

    summary(model, (3, 224, 224))