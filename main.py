from __future__ import annotations
from keras.datasets import mnist
from keras.utils import to_categorical
from src.genetic_algorithm import GeneticAlgorithm


def main():
    """This project proposes the use of a Genetic Algorithm to discover
    an optimized model of a convolutional network for the MNIST dataset.
    This process is conducted using the Pareto frontier and the
    lexicographic approach. The user has the freedom to select the
    metrics to be optimized and define their respective tolerance
    thresholds."""
    (
        (train_images, train_labels),
        (test_images, test_labels),
    ) = mnist.load_data()

    # Split into a small dataset due to limited resources
    train_images = train_images[:7000]
    test_images = test_images[:3000]
    train_labels = train_labels[:7000]
    test_labels = test_labels[:3000]

    train_images = (
        train_images.reshape((7000, 28, 28, 1)).astype("float32") / 255
    )
    test_images = (
        test_images.reshape((3000, 28, 28, 1)).astype("float32") / 255
    )

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

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
        second_objective=second_objective
    )

    # Execution of genetic algorithm
    result = ga.solve(generations=10)

    print(result)

    result.model.save(f"{first_objective[0]}_{second_objective[0]}.h5")


if __name__ == "__main__":
    main()
