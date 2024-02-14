from __future__ import annotations
import numpy as np
from src.individual import Individual
from tqdm import tqdm
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, population_size: int, X_train, X_test, y_train, y_test, first_objective, frist_threshold, second_objective, second_threshold) -> None:
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
        self.fo = first_objective
        self.fo_threshold = frist_threshold
        self.so = second_objective
        self.so_threshold = second_threshold

    def init_population(self) -> None:
        for _ in range(self.population_size):
            self.population.append(Individual())
        self.best_solution = self.population[0]

    def sort_population(self) -> None:
        self.population = sorted(self.population, key=lambda individual: individual.metrics[self.fo][0]*individual.metrics[self.fo][1], reverse=True)

    def best_individual(self, pareto) -> Individual:
        first_minimum = min(pareto, key=lambda x: x.metrics[self.fo][0]*x.metrics[self.fo][1])
        first_threshold = first_minimum.metrics[self.fo][1] * self.fo_threshold
        low_bound = first_minimum.metrics[self.fo][0]*first_minimum.metrics[self.fo][1] - first_threshold
        high_bound = first_minimum.metrics[self.fo][0]*first_minimum.metrics[self.fo][1] + first_threshold
        interval = [
            individual for individual in pareto if low_bound <= individual.metrics[self.fo][0]*individual.metrics[self.fo][1] <= high_bound
        ]
        second_minimum = min(interval, key=lambda x: x.metrics[self.so][0]*x.metrics[self.so][1])
        second_threshold = second_minimum.metrics[self.so][1] * self.so_threshold
        low_bound = second_minimum.metrics[self.so][0]*second_minimum.metrics[self.so][1] - second_threshold
        high_bound = second_minimum.metrics[self.so][0]*second_minimum.metrics[self.so][1] + second_threshold
        interval2 = [
            individual for individual in interval if low_bound <= individual.metrics[self.so][0]*individual.metrics[self.so][1] <= high_bound
        ]
        if len(interval2) == 1:
            return second_minimum
        else:
            return min(interval2, key=lambda x: x.metrics[self.fo][0]*x.metrics[self.fo][1])

    def sum_eval(self) -> float:
        sum = 0
        for individual in self.population:
            sum += individual.metrics["accuracy"][1]
        return sum

    def select_parent(self, size:int) -> int:
        return np.random.randint(size)

    def visualize_generation(self, best) -> None:
        print(f"G:{best.generation}:\n",
              f"{self.fo}: {round(best.metrics[self.fo][1], 2)}\n",
              f"{self.so}: {round(best.metrics[self.so][1], 2)}\n",
              f" Gene: {best.gene}")

    def pareto_result(self):
        frontier_fo = []
        frontier_so = []
        for individual in self.pareto_frontier:
            frontier_fo.append(individual.metrics[self.fo][0]*individual.metrics[self.fo][1])
            frontier_so.append(individual.metrics[self.so][0]*individual.metrics[self.so][1])
        pareto_fo = []
        pareto_so = []
        for individual in self.frontier:
            pareto_fo.append(individual.metrics[self.fo][0]*individual.metrics[self.fo][1])
            pareto_so.append(individual.metrics[self.so][0]*individual.metrics[self.so][1])
        self.best_solution = self.best_individual(self.pareto_frontier)
        plt.figure(figsize=(10,6), dpi=800)
        plt.plot(pareto_so, pareto_fo, color="blue", marker="*", linestyle="None", label="Dominated Solutions")
        plt.plot(frontier_so, frontier_fo, color="red", marker="*", linestyle="None", label="Non Dominated Solutions")
        plt.plot(
            self.best_solution.metrics[self.so][0]*self.best_solution.metrics[self.so][1],
            self.best_solution.metrics[self.fo][0]*self.best_solution.metrics[self.fo][1],
            color="green",
            marker="*",
            linestyle="None",
            label="Chosen Solution"
        )
        plt.xlabel(f"{self.so}")
        plt.ylabel(f"{self.fo}")
        plt.title(f"{self.fo} x {self.so}")
        plt.legend()

    def solve(self, mutation_rate=0.05, generations=0) -> list[int]:
        self.init_population()

        print("Generation 0")

        for i in tqdm(range(len(self.population))):
            self.population[i]._evaluate(self.X_train, self.X_test, self.y_train, self.y_test)


        for i in range(generations):
            for individual in self.population:
                frontier = not any(
                    (
                        individual.metrics[self.fo][0] * individual.metrics[self.fo][1] >
                        other_individual.metrics[self.fo][0] * other_individual.metrics[self.fo][1] and
                        individual.metrics[self.so][0] * individual.metrics[self.so][1] >
                        other_individual.metrics[self.so][0] * other_individual.metrics[self.so][1]
                    ) for other_individual in self.population
                )

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
                    self.best_solution = self.best_individual(self.pareto)
            self.visualize_generation(self.best_solution)
            print(f"Generation {i + 1}")

            self.population = list(new_population)

            for i in tqdm(range(len(self.population))):
                self.population[i]._evaluate(self.X_train, self.X_test, self.y_train, self.y_test)

        self.frontier = [item for row in self.frontier for item in row]

        self.pareto_frontier = []
        for individual in self.frontier:
            frontier = not any(
                (
                    individual.metrics[self.fo][0] * individual.metrics[self.fo][1] >
                    other_individual.metrics[self.fo][0] * other_individual.metrics[self.fo][1] and
                    individual.metrics[self.so][0] * individual.metrics[self.so][1] >
                    other_individual.metrics[self.so][0] * other_individual.metrics[self.so][1]
                ) for other_individual in self.frontier
            )

            if frontier:
                self.pareto_frontier.append(individual)
        self.pareto_result()

        return self.best_solution