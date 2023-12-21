from sklearn.metrics import mean_squared_error

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from typing import *

import numpy as np

import cv2

import os

def image_pca(image : np.ndarray, n_components : int) -> np.ndarray:

    assert isinstance(image, np.ndarray)

    assert isinstance(n_components, int)

    assert np.prod(image.shape) > 0

    assert n_components > 0
    
    pca = PCA(n_components = n_components)

    gray_image_normalized = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0

    reduced_image = pca.fit_transform(gray_image_normalized)

    reconstructed_image = pca.inverse_transform(reduced_image)

    reconstructed_image = cv2.normalize(reconstructed_image, None, 0, 255, cv2.NORM_MINMAX)

    # reconstructed_image *= 255.0

    return np.clip(reconstructed_image, 0, 255).astype(np.uint8)

def find_optimal_component_n(image : np.ndarray) -> int:

    assert isinstance(image, np.ndarray)

    assert np.prod(image.shape) > 0

    assert image.shape.__len__() == 3

    assert image.shape[2] == 3

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_image_normalized = gray_image / 255.0

    h, w, *_ = gray_image.shape

    n_components_range = min(h, w)

    min_error = float('inf')

    min_n_components = 0

    counter = 1

    for n_components in reversed(range(1, n_components_range + 1)):

        pca = PCA(n_components = n_components)

        reduced_image = pca.fit_transform(gray_image_normalized)
        reconstructed_image = pca.inverse_transform(reduced_image)

        # reconstructed_image *= 255.0

        reconstructed_image = cv2.normalize(reconstructed_image, None, 0, 255, cv2.NORM_MINMAX)

        mse = mean_squared_error(gray_image.reshape((-1, )), reconstructed_image.astype(int).reshape((-1,)))

        counter += 1

        if mse <= 3.0:
            if (mse < min_error):
                min_error = mse
            min_n_components = n_components
        else:
            break

    return min_n_components

if (__name__ == "__main__"):

    image_path = os.path.join(os.path.dirname(__file__), "data/Q3/logo.jpg")

    rgb_image = cv2.imread(image_path)

    min_n_components = find_optimal_component_n(rgb_image)

    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    gray_image_normalized = gray_image / 255.0

    print("Minimum number of components with error <= 3.0:", min_n_components)

    reconstructed_image = image_pca(rgb_image, min_n_components)

    plt.figure(figsize = (8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap = 'gray')
    plt.title('Original Gray Image')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap = 'gray')
    plt.title(f'Reconstructed Image (n={min_n_components})')

    plt.show()
