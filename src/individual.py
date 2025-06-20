from __future__ import annotations
import numpy as np
import keras
from sklearn.metrics import f1_score, roc_auc_score
from time import time
import tensorflow as tf


CONV_LAYERS_GENES = {
    "0": 3,
    "1": 4
}

NEURONS_CONV_GENES = {
    "000": 32,
    "001": 64,
    "010": 96,
    "011": 128,
    "100": 192,
    "101": 256,
    "110": 384,
    "111": 512
}

DROPOUT_GENES = {
    "00": 0,
    "01": 0.25,
    "10": 0.50,
    "11": 0.75
}

NEURONS_DENSE_GENES = {
    "00": 32,
    "01": 64,
    "10": 128,
    "11": 286
}

GENOME_SIZE = 8
CONV_LAYERS_SLICE = slice(0, 1)
NEURONS_CONV_SLICE = slice(1, 4)
DROPOUT_SLICE = slice(4, 6)
NEURONS_DENSE_SLICE = slice(6, 8)
NUM_CLASSES = 9


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
        self.genome = np.random.randint(0, 2, GENOME_SIZE).tolist()
        self.metrics = {}

    def _decode_gene(self, gene_map: dict, gene_range: slice) -> int:
        """_summary_

        Args:
            gene_range (slice): _description_

        Returns:
            int: _description_
        """
        return gene_map["".join(map(str, self.genome[gene_range]))]

    def _evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
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
        # Set seed for reproductibility of the models with the same genome
        seed = int("".join(map(str, self.genome)), 2)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.model = keras.Sequential()
        self.model.add(
            keras.layers.Conv2D(
                self._decode_gene(NEURONS_CONV_GENES, NEURONS_CONV_SLICE),  # Maps the number of neurons of all conv layers
                (3, 3),
                activation="relu",
                input_shape=(28, 28, 3),
                padding="same"
            )
        )
        for _ in range(self._decode_gene(CONV_LAYERS_GENES, CONV_LAYERS_SLICE)):
            self.model.add(
                keras.layers.Conv2D(
                    self._decode_gene(NEURONS_CONV_GENES, NEURONS_CONV_SLICE),  
                    (3, 3),
                    activation="relu",
                    padding="same"
                )
            )
            self.model.add(keras.layers.MaxPooling2D((2, 2)))
            self.model.add(keras.layers.Dropout(self._decode_gene(DROPOUT_GENES, DROPOUT_SLICE)))

        self.model.add(keras.layers.Flatten())

        self.model.add(
            keras.layers.Dense(
                self._decode_gene(NEURONS_DENSE_GENES, NEURONS_DENSE_SLICE),  # Maps the number neurons in the dense layers
                activation="relu",
            )
        )
        self.model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        self.model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=16,
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
        y_prob = self.model.predict(X_test, verbose=False)
        end = time()
        y_pred = np.argmax(y_prob, axis=1)
        y_test = np.argmax(y_test, axis=1)
        self.metrics["f1_score"] = (
            -1,
            f1_score(y_test, y_pred, average="macro"),
        )

        # Latency is calculated by the avarege execution time
        self.metrics["latency"] = (1, (end - start) / 3000)

        try:
            auc_score = roc_auc_score(y_test, y_prob, multi_class="ovr")
        except ValueError:
            auc_score = 0.0  # Se não for possível calcular, assume-se 0

        self.metrics["auc"] = (-1, auc_score)

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
{self._decode_gene(CONV_LAYERS_GENES, CONV_LAYERS_SLICE)} camadas convolucionais com {self._decode_gene(NEURONS_CONV_GENES, NEURONS_CONV_SLICE)} neurons
1 camada densa com {self._decode_gene(NEURONS_DENSE_GENES, NEURONS_DENSE_SLICE)} neurons
"""
