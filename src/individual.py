from __future__ import annotations
import numpy as np
import keras
from sklearn.metrics import f1_score
from time import time

class Individual:
    def __init__(self, generation: int=0) -> None:
        self.generation = generation
        self.gene = [0 if np.random.random() < 0.5 else 1 for _ in range(17)]
        self.eval = 0
        self.metrics = {}

    def _evaluate(self, X_train, X_test, y_train, y_test) -> None:
        self.model = keras.Sequential()
        self.model.add(keras.layers.Conv2D(2 ** (int("".join(map(str, self.gene[2:4])), 2) + 4), (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(keras.layers.MaxPooling2D((2, 2)))
        for i in range(1, int("".join(map(str, self.gene[0:2])), 2)):
            self.model.add(keras.layers.Conv2D(2 ** (int("".join(map(str, self.gene[2:4])), 2) + 4), (3, 3), activation="relu"))
            if (i % 2) != 0:
                self.model.add(keras.layers.MaxPooling2D((2, 2)))
        self.model.add(keras.layers.Flatten())
        for _ in range(int("".join(map(str, self.gene[9:11])), 2) + 1):
            self.model.add(keras.keras.layers.Dense(int("".join(map(str, self.gene[11:17])), 2) + 1, activation="relu"))
        self.model.add(keras.layers.Dense(10, activation='softmax'))

        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=False)
        weights = self.model.get_weights()
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=False)
        self.metrics["loss"] = (1, loss)
        self.metrics["accuracy"] = (-1, accuracy)
        self.metrics["weights_norm"]= (1, sum([np.linalg.norm(w) for w in weights]))
        start = time()
        y_pred = self.model.predict(X_test, verbose=False)
        end = time()
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        self.metrics["f1_score"] = (-1, f1_score(y_test, y_pred, average="macro"))
        self.metrics["latency"] = (1, (end - start) / 3000)


    def crossover(self, other_individual: Individual) -> list[Individual]:
        cut = round(np.random.rand() * len(self.gene))
        child1 = other_individual.gene[0:cut] + self.gene[cut::]
        child2 = self.gene[0:cut] + other_individual.gene[cut::]

        children = [Individual(self.generation + 1), Individual(self.generation + 1)]

        children[0].gene = child1
        children[1].gene = child2

        return children

    def mutation(self, mutation_rate: float) -> Individual:
        for i in range(len(self.gene)):
            if np.random.random() < mutation_rate:
                if self.gene[i]:
                    self.gene[i] = 0
                else:
                    self.gene[i] = 1
        return self

    def __repr__(self):
        return f"""Generation: {self.generation}
Gene: {self.gene}
{int("".join(map(str, self.gene[0:2])), 2) + 1} camadas convolucionais com {int("".join(map(str, self.gene[2:9])), 2) + 1} neurons
{int("".join(map(str, self.gene[9:11])), 2) + 1} camadas densas com {int("".join(map(str, self.gene[11:17])), 2) + 1} neurons
"""