from __future__ import annotations
import numpy as np
import keras

class Individual:
    def __init__(self, generation: int=0) -> None:
        self.generation = generation
        self.gene = [0 if np.random.random() < 0.5 else 1 for _ in range(12)]
        self.eval = 0
    
    def _evaluate(self, X_train, X_test, y_train, y_test) -> None:
        self.model = keras.Sequential()
        self.model.add(keras.layers.Input(shape=(20, )))
        for _ in range(int("".join(map(str, self.gene[0:3])), 2) + 1):
            self.model.add(keras.layers.Dense(int("".join(map(str, self.gene[3:7])), 2) + 1, activation="relu"))
        for _ in range(int("".join(map(str, self.gene[7:9])), 2) + 1):
            self.model.add(keras.layers.Dense(int("".join(map(str, self.gene[9:12])), 2) + 1, activation="relu"))
        self.model.add(keras.layers.Dense(1, activation="sigmoid"))

        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2, verbose=False)

        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=False)

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
{int("".join(map(str, self.gene[0:3])), 2) + 1} camadas com {int("".join(map(str, self.gene[3:7])), 2) + 1} neurons
{int("".join(map(str, self.gene[7:9])), 2) + 1} camadas com {int("".join(map(str, self.gene[9:12])), 2) + 1} neurons
"""