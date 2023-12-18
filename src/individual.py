from __future__ import annotations
import numpy as np
import keras

class Individual:
    def __init__(self, generation: int=0) -> None:
        self.generation = generation
        self.gene = [0 if np.random.random() < 0.5 else 1 for _ in range(17)]
        self.eval = 0

    def _evaluate(self, train_images, test_images, train_labels, test_labels) -> None:
        self.model = keras.Sequential()
        self.model.add(keras.layers.Conv2D(2 ** (int("".join(map(str, self.gene[2:4])), 2) + 4), (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(keras.layers.MaxPooling2D((2, 2)))
        for i in range(1, int("".join(map(str, self.gene[0:2])), 2)):
            self.model.add(keras.layers.Conv2D(2 ** (int("".join(map(str, self.gene[2:4])), 2) + 4), (3, 3), activation="relu"))
            if (i % 2) != 0:
                self.model.add(keras.layers.MaxPooling2D((2, 2)))
        self.model.add(keras.layers.Flatten())
        for _ in range(int("".join(map(str, self.gene[9:11])), 2) + 1):
            self.model.add(keras.layers.Dense(int("".join(map(str, self.gene[11:17])), 2) + 1, activation="relu"))
        self.model.add(keras.layers.Dense(10, activation='softmax'))

        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)

        loss, accuracy = self.model.evaluate(test_images, test_labels, verbose=False)

        self.loss = loss
        self.eval = accuracy

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