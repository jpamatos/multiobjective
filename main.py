from __future__ import annotations
from keras.datasets import mnist
from keras.utils import to_categorical
from src.genetic_algorithm import GeneticAlgorithm
from sklearn.model_selection import train_test_split
from typing import Tuple
import numpy as np


def get_images() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and preprocesses the MNIST dataset.

    Returns:
        tuple: A tuple containing four elements:
            - train_images (numpy.ndarray): An array containing the training images.
            - train_labels (numpy.ndarray): An array containing the labels corresponding to the training images.
            - test_images (numpy.ndarray): An array containing the test images.
            - test_labels (numpy.ndarray): An array containing the labels corresponding to the test images.
    """
    (
        (train_images, train_labels),
        (test_images, test_labels),
    ) = mnist.load_data()

    train_images, _, train_labels, _ = train_test_split(
        train_images,
        train_labels,
        train_size=7000,
        stratify=train_labels,
        random_state=1,
    )
    test_images, _, test_labels, _ = train_test_split(
        test_images,
        test_labels,
        train_size=3000,
        stratify=test_labels,
        random_state=1,
    )

    train_images = (
        train_images.reshape((7000, 28, 28, 1)).astype("float32") / 255
    )
    test_images = (
        test_images.reshape((3000, 28, 28, 1)).astype("float32") / 255
    )

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

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
    second_objective = ("latency", 0.15)

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
