from __future__ import annotations
import numpy as np
from src.individual import Individual
from tqdm import tqdm

class GeneticAlgorithm:
    def __init__(self, population_size: int, X_train, X_test, y_train, y_test) -> None:
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.eval_sum = 0
        self.best_solution = None
        self.solution_list = []
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.pareto = []
        self.frontier = []

    def init_population(self) -> None:
        for _ in range(self.population_size):
            self.population.append(Individual())
        self.best_solution = self.population[0]

    def sort_population(self) -> None:
        self.population = sorted(self.population, key=lambda individual: individual.eval, reverse=True)

    def best_individual(self) -> Individual:
        maximum = max(self.pareto, key=lambda x: x.eval)
        threshold_first = maximum.eval * 0.05
        interval = [individual for individual in self.pareto if maximum.eval - threshold_first <= maximum.eval <= maximum.eval + threshold_first]
        minimum = min(interval, key=lambda x:x.loss)
        threshold_second = minimum.loss * 0.05
        interval2 = [individual for individual in interval if minimum.loss - threshold_second <= minimum.loss <= minimum.loss + threshold_second]
        if len(interval2) == 1:
            return minimum
        else:
            return max(interval2, key=lambda x: x.eval)

    def sum_eval(self) -> float:
        sum = 0
        for individual in self.population:
            sum += individual.eval
        return sum

    def select_parent(self, size:int) -> int:
        return np.random.randint(size)

    def visualize_generation(self) -> None:
        best = self.population[0]
        print(f"G:{self.population[0].generation} -> ",
              f"Eval: {round(best.eval, 2)}",
              f" Gene: {best.gene}")

    def solve(self, mutation_rate=0.05, generations=0) -> list[int]:
        self.init_population()

        print("Generation 0")

        for i in tqdm(range(len(self.population))):
            self.population[i]._evaluate(self.X_train, self.X_test, self.y_train, self.y_test)


        for i in range(generations):
            for individual in self.population:
                frontier = not any((individual.loss > other_individual.loss and individual.eval < other_individual.eval) for other_individual in self.population)

                if frontier:
                    self.pareto.append(individual)
            self.frontier.append(self.pareto)
            new_population = []
            if len(self.pareto) == 1:
                for _ in range(0, self.population_size, 2):
                    parent1 = self.select_parent(len(self.pareto))
                    parent2 = self.select_parent(self.population_size)

                    children = self.pareto[parent1].crossover(self.population[parent2])

                    new_population.append(children[0].mutation(mutation_rate))
                    new_population.append(children[1].mutation(mutation_rate))
                    self.best_solution = self.pareto[0]
            else:
                for _ in range(0, self.population_size, 2):
                    parent1 = self.select_parent(len(self.pareto))
                    parent2 = self.select_parent(len(self.pareto))

                    children = self.pareto[parent1].crossover(self.pareto[parent2])

                    new_population.append(children[0].mutation(mutation_rate))
                    new_population.append(children[1].mutation(mutation_rate))
                    self.best_solution = self.best_individual()
            print(f"Best Solution -> G:{self.best_solution.generation} -> ",
              f"Eval: {round(self.best_solution.eval, 3)}",
              f" Gene: {self.best_solution.gene}")
            print(f"Generation {i + 1}")

            self.population = list(new_population)

            for i in tqdm(range(len(self.population))):
                self.population[i]._evaluate(self.X_train, self.X_test, self.y_train, self.y_test)

        self.frontier = [item for row in self.frontier for item in row]

        self.pareto_frontier = []
        for individual in self.frontier:
            frontier = not any((individual.loss > other_individual.loss and individual.eval < other_individual.eval) for other_individual in self.frontier)

            if frontier:
                self.pareto_frontier.append(individual)

        return self.best_solution