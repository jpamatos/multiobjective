from __future__ import annotations
import numpy as np
import keras
from sklearn.metrics import f1_score
from time import time


class Individual:
    """A class that represents the individual of the population, enconding
        parameters of a neural network.

    Attributes:
        generation (int): Represents which generation this individual was
            created.
        genome (list): The genome of the individual, encoding parameters
            for the neural network.
        metrics (dict): The metrics of the evaluated neural network
            (accuracy, loss, f1_score, latency, weights_norm).
        model (keras.Sequential): Neural Network Model created by the
            genome encoded properties.
    """

    def __init__(self, generation: int = 0) -> None:
        """
        Initialize the Individual with its genome, generation and metrics.

        Args:
            genome (list): The genome of the individual.
            fitness (float): The fitness score of the individual.
        """
        self.generation = generation
        self.genome = [0 if np.random.random() < 0.5 else 1 for _ in range(17)]
        self.metrics = {}

    def _evaluate(self, X_train, X_test, y_train, y_test) -> None:
        """
        Evaluate the individual's neural network model using the provided data.

        This method constructs a neural network based on the individual's genome
        and evaluates its performance using the given training and testing datasets.

        Args:
            X_train (numpy.ndarray): Training input data.
            X_test (numpy.ndarray): Testing input data.
            y_train (numpy.ndarray): Training target data (labels).
            y_test (numpy.ndarray): Testing target data (labels).

        Returns:
            None

        Example:
            individual._evaluate(X_train, X_test, y_train, y_test)
        """
        self.model = keras.Sequential()
        self.model.add(
            keras.layers.Conv2D(
                2
                ** (
                    int("".join(map(str, self.genome[2:4])), 2) + 4
                ),  # Maps the number of neurons in the first conv layer
                (3, 3),
                activation="relu",
                input_shape=(28, 28, 1),
            )
        )
        self.model.add(keras.layers.MaxPooling2D((2, 2)))
        for i in range(
            1, int("".join(map(str, self.genome[0:2])), 2)
        ):  # Maps the number of conv layers
            self.model.add(
                keras.layers.Conv2D(
                    2
                    ** (
                        int("".join(map(str, self.genome[2:4])), 2) + 4
                    ),  # Maps the number of neurons of all conv layers
                    (3, 3),
                    activation="relu",
                )
            )
            # Create max pool layers after a odd number of layers
            if (i % 2) != 0:
                self.model.add(keras.layers.MaxPooling2D((2, 2)))

        self.model.add(keras.layers.Flatten())
        for _ in range(
            int("".join(map(str, self.genome[9:11])), 2) + 1
        ):  # Maps the number of dense layers
            self.model.add(
                keras.keras.layers.Dense(
                    int("".join(map(str, self.genome[11:17])), 2)
                    + 1,  # Maps the number neurons in the dense layers
                    activation="relu",
                )
            )
        self.model.add(keras.layers.Dense(10, activation="softmax"))

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        self.model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=64,
            validation_split=0.2,
            verbose=False,
        )
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=False)
        self.metrics["loss"] = (1, loss)
        self.metrics["accuracy"] = (-1, accuracy)

        # Get the weights on the trainable layers
        weights = self.model.get_weights()
        self.metrics["weights_norm"] = (
            1,
            sum([np.linalg.norm(w) for w in weights]),
        )
        start = time()
        y_pred = self.model.predict(X_test, verbose=False)
        end = time()
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        self.metrics["f1_score"] = (
            -1,
            f1_score(y_test, y_pred, average="macro"),
        )

        # Latency is calculated by the avarege execution time
        self.metrics["latency"] = (1, (end - start) / 3000)

    def crossover(self, other_individual: Individual) -> list[Individual]:
        """Performs crossover between the current individual and another individual.

        This method combines genetic material from the current individual and
        the provided other_individual to generate new children individuals.

        Args:
            other_individual (Individual): the second individual to crossover with

        Returns:
            list[Individual]: list containing the two individuals from the crossover

        Example:
            children = individual.crossover(other_individual)
        """
        # Performs a single cut in a random gene on the genome
        cut = round(np.random.rand() * len(self.genome))
        child1 = other_individual.genome[0:cut] + self.genome[cut::]
        child2 = self.genome[0:cut] + other_individual.genome[cut::]

        # Represents the advancement of the generation 
        children = [
            Individual(self.generation + 1),
            Individual(self.generation + 1),
        ]
        children[0].genome = child1
        children[1].genome = child2

        return children

    def mutation(self, mutation_rate: float) -> Individual:
        """Apply mutation to the individual's genome.
        Apply mutation to the individual.

        Args:
            mutation_rate (float): The probability of mutation for each gene.

        Returns:
            Individual: A new individual with mutations applied.

        Example:
            individual_mutated = individual.mutation(0.05)
        """
        for i in range(len(self.genome)):
            if np.random.random() < mutation_rate:
                if self.genome[i]:
                    self.genome[i] = 0
                else:
                    self.genome[i] = 1
        return self

    def __repr__(self):
        return f"""Generation: {self.generation}
Gene: {self.genome}
{int("".join(map(str, self.genome[0:2])), 2) + 1} camadas convolucionais com {int("".join(map(str, self.genome[2:9])), 2) + 1} neurons
{int("".join(map(str, self.genome[9:11])), 2) + 1} camadas densas com {int("".join(map(str, self.genome[11:17])), 2) + 1} neurons
"""
