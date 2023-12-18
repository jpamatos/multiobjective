from __future__ import annotations
from keras.datasets import mnist
from keras.utils import to_categorical
from src.genetic_algorithm import GeneticAlgorithm
from src.individual import Individual

def main():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images[:7000]
    test_images = test_images[:3000]
    train_labels = train_labels[:7000]
    test_labels = test_labels[:3000]

    train_images = train_images.reshape((7000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((3000, 28, 28, 1)).astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    ga = GeneticAlgorithm(6, train_images, test_images, train_labels, test_labels)

    result = ga.solve(generations=10)

    print(result)

if __name__ == "__main__":
    main()