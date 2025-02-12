from __future__ import annotations

from typing import Tuple
import warnings

import numpy as np
from keras import utils
from medmnist import PathMNIST

from src.genetic_algorithm import GeneticAlgorithm

warnings.filterwarnings("ignore")


def get_images() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and preprocesses the MNIST dataset.

    Returns:
        tuple: A tuple containing four elements:
            - train_images (numpy.ndarray): An array containing the training images.
            - train_labels (numpy.ndarray): An array containing the labels corresponding to the training images.
            - test_images (numpy.ndarray): An array containing the test images.
            - test_labels (numpy.ndarray): An array containing the labels corresponding to the test images.
    """
    dataset = PathMNIST(split="train", download=True)
    test_dataset = PathMNIST(split="test", download=True)

    train_images, train_labels = dataset.imgs, dataset.labels
    test_images, test_labels = test_dataset.imgs, test_dataset.labels

    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    train_images = train_images.reshape((-1, 28, 28, 3))  
    test_images = test_images.reshape((-1, 28, 28, 3))

    num_classes = len(np.unique(train_labels))
    train_labels = utils.to_categorical(train_labels, num_classes)
    test_labels = utils.to_categorical(test_labels, num_classes)

    return train_images, train_labels, test_images, test_labels


def main() -> None:
    """This project proposes the use of a Genetic Algorithm to discover
    an optimized model of a convolutional network for the MNIST dataset.
    This process is conducted using the Pareto frontier and the
    lexicographic approach. The user has the freedom to select the
    metrics to be optimized and define their respective tolerance
    thresholds."""

    train_images, train_labels, test_images, test_labels = get_images()

    # User chosen objectives
    first_objective = ("accuracy", 0.05)
    second_objective = ("auc", 0.15)

    ga = GeneticAlgorithm(
        population_size=12,
        X_train=train_images,
        X_test=test_images,
        y_train=train_labels,
        y_test=test_labels,
        first_objective=first_objective,
        second_objective=second_objective,
    )

    # Execution of genetic algorithm
    result = ga.solve(generations=10)

    print(result)

    result.model.save(f"{first_objective[0]}_{second_objective[0]}.keras")


if __name__ == "__main__":
    main()
