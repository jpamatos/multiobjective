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
        self.best_solution = 0
        self.solution_list = []
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.pareto = []

    def init_population(self) -> None:
        for _ in range(self.population_size):
            self.population.append(Individual())
        self.best_solution = self.population[0]

    def sort_population(self) -> None:
        self.population = sorted(self.population, key=lambda individual: individual.eval, reverse=True)

    def best_individual(self, individual: Individual) -> None:
        if individual.eval > self.best_solution.eval:
            self.best_solution = individual
    
    def sum_eval(self) -> float:
        sum = 0
        for individual in self.population:
            sum += individual.eval
        return sum

    def select_parent(self, sum_eval:float) -> int:
        parent = -1
        value = np.random.rand() * sum_eval
        sum = 0
        i = 0
        while i < len(self.population) and sum < value:
            sum += self.population[i].eval
            parent += 1
            i += 1
        return parent

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
        
        self.sort_population()

        self.best_solution = self.population[0]
        self.solution_list.append(self.best_solution.eval)

        self.visualize_generation()

        for i in range(generations):
            print(f"Generation {i + 1}")
            eval_sum = self.sum_eval()
            new_population = []

            for _ in range(0, self.population_size, 2):
                parent1 = self.select_parent(eval_sum)
                parent2 = self.select_parent(eval_sum)

                children = self.population[parent1].crossover(self.population[parent2])

                new_population.append(children[0].mutation(mutation_rate))   
                new_population.append(children[1].mutation(mutation_rate))

            self.population = list(new_population)

            for i in tqdm(range(len(self.population))):
                self.population._evaluate(self.X_train, self.X_test, self.y_train, self.y_test)
            
            pareto = []
            for individual in self.population:
                frontier = not any((individual.loss > other_individual.loss and individual.eval < other_individual.eval) for other_individual in self.population)

                if frontier:
                    pareto.append(individual)

            self.pareto.extend(pareto)
            

            self.sort_population()
            self.visualize_generation()

            self.solution_list.append(self.population[0].eval)

            self.best_individual(self.population[0])
        
        self.pareto_frontier = []
        for individual in self.pareto:
            frontier = not any((individual.loss > other_individual.loss and individual.eval < other_individual.eval) for other_individual in self.pareto)

            if frontier:
                self.pareto_frontier.append(individual)

        print(f"Best Solution -> G:{self.best_solution.generation} -> ",
              f"Eval: {round(self.best_solution.eval, 3)}",
              f" Gene: {self.best_solution.gene}")
        
        return self.best_solution