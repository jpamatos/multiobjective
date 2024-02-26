import numpy as np
from src.individual import Individual
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple


class GeneticAlgorithm:
    """A class representing a genetic algorithm for optimization of Neural Networks.

    Attributes:
        population_size (int): The size of the population.
        population (list): The list of individuals in the population.
        generation (int): The current generation number.
        best_solution (Individual): The best solution found.
        X_train: Training data for the genetic algorithm.
        X_test: Testing data for the genetic algorithm.
        y_train: Training labels for the genetic algorithm.
        y_test: Testing labels for the genetic algorithm.
        pareto (list): A list of individuals representing the Pareto frontier.
        frontier (list): A list of fronts of non-dominated solutions.
        fo (str): The name of the first objective.
        fo_threshold (float): The threshold for the first objective.
        so (str): The name of the second objective.
        so_threshold (float): The threshold for the second objective.
    """

    def __init__(
        self,
        population_size: int,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        first_objective: Tuple[str, int],
        second_objective: Tuple[str, int],
    ) -> None:
        """Initializes a GeneticAlgorithm instance with the provided parameters.

        Args:
            population_size (int): The size of the population.
            X_train: Training data for the genetic algorithm.
            X_test: Testing data for the genetic algorithm.
            y_train: Training labels for the genetic algorithm.
            y_test: Testing labels for the genetic algorithm.
            first_objective: A tuple containing the name and threshold for the first objective.
            second_objective: A tuple containing the name and threshold for the second objective.

        Returns:
            None
        """
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.best_solution = None
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.pareto = []
        self.frontier = []
        self.fo = first_objective[0]
        self.fo_threshold = first_objective[1]
        self.so = second_objective[0]
        self.so_threshold = second_objective[1]

    def init_population(self) -> None:
        """Initializes the population of individuals.

        Returns:
            None
        """
        for _ in range(self.population_size):
            self.population.append(Individual())
        self.best_solution = self.population[0]

    def sort_population(self) -> None:
        """Sorts the population based on a specified fitness metric.

        Returns:
            None
        """
        self.population = sorted(
            self.population,
            key=lambda individual: individual.metrics[self.fo][0]
            * individual.metrics[self.fo][1],
            reverse=True,
        )

    def best_individual(self, pareto: list[Individual]) -> Individual:
        """Finds the best individual from a Pareto front based on specified fitness metrics.

        Args:
            pareto (list[Individual]): List of individuals representing the Pareto front.

        Returns:
            Individual: The best individual from the Pareto front.
        """
        first_minimum = min(
            pareto, key=lambda x: x.metrics[self.fo][0] * x.metrics[self.fo][1]
        )

        # Threshold is min + threshold(%)
        first_threshold = first_minimum.metrics[self.fo][1] * self.fo_threshold
        low_bound = (
            first_minimum.metrics[self.fo][0]
            * first_minimum.metrics[self.fo][1]
            - first_threshold
        )
        high_bound = (
            first_minimum.metrics[self.fo][0]
            * first_minimum.metrics[self.fo][1]
            + first_threshold
        )

        # Subset containing the solutions between the minimum of the first objective and its threshold
        interval = [
            individual
            for individual in pareto
            if low_bound
            <= individual.metrics[self.fo][0] * individual.metrics[self.fo][1]
            <= high_bound
        ]
        second_minimum = min(
            interval,
            key=lambda x: x.metrics[self.so][0] * x.metrics[self.so][1],
        )
        second_threshold = (
            second_minimum.metrics[self.so][1] * self.so_threshold
        )
        low_bound = (
            second_minimum.metrics[self.so][0]
            * second_minimum.metrics[self.so][1]
            - second_threshold
        )
        high_bound = (
            second_minimum.metrics[self.so][0]
            * second_minimum.metrics[self.so][1]
            + second_threshold
        )

        # Subset containing the solutions between the minimum of the second objective and its threshold
        interval2 = [
            individual
            for individual in interval
            if low_bound
            <= individual.metrics[self.so][0] * individual.metrics[self.so][1]
            <= high_bound
        ]
        if len(interval2) == 1:
            return second_minimum
        else:
            return min(
                interval2,
                key=lambda x: x.metrics[self.fo][0] * x.metrics[self.fo][1],
            )

    def select_parent(self, size: int) -> int:
        """Selects a parent index randomly from the range [0, size).

        Args:
            size (int): The size of the population.

        Returns:
            int: The index of the selected parent.
        """
        return np.random.randint(size)

    def visualize_generation(self, best: Individual) -> None:
        """Visualizes the best individual's information for a specific generation.

        Prints the generation number, fitness metrics, and genome of the best individual.

        Args:
            best (Individual): The best individual in the generation.

        Returns:
            None
        """
        print(
            f"G:{best.generation}:\n",
            f"{self.fo}: {round(best.metrics[self.fo][1], 2)}\n",
            f"{self.so}: {round(best.metrics[self.so][1], 2)}\n",
            f" Genome: {best.genome}",
        )

    def pareto_result(self):
        """Visualizes the Pareto frontier and the best solution.

        Calculates the fitness values for individuals in the Pareto frontier and the frontier.
        Plots the Pareto frontier and the frontier in a 2D plot along with the best solution.

        Returns:
            None
        """
        frontier_fo = []
        frontier_so = []

        # Separate the non dominated solutions from the dominated solutions
        for individual in self.pareto_frontier:
            frontier_fo.append(
                individual.metrics[self.fo][0] * individual.metrics[self.fo][1]
            )
            frontier_so.append(
                individual.metrics[self.so][0] * individual.metrics[self.so][1]
            )
        pareto_fo = []
        pareto_so = []
        for individual in self.frontier:
            pareto_fo.append(
                individual.metrics[self.fo][0] * individual.metrics[self.fo][1]
            )
            pareto_so.append(
                individual.metrics[self.so][0] * individual.metrics[self.so][1]
            )
        self.best_solution = self.best_individual(self.pareto_frontier)
        plt.figure(figsize=(10, 6), dpi=800)
        plt.plot(
            pareto_so,
            pareto_fo,
            color="blue",
            marker="*",
            linestyle="None",
            label="Dominated Solutions",
        )
        plt.plot(
            frontier_so,
            frontier_fo,
            color="red",
            marker="*",
            linestyle="None",
            label="Non Dominated Solutions",
        )

        # Show the best solution of the genetic algorithm (in green)
        plt.plot(
            self.best_solution.metrics[self.so][0]
            * self.best_solution.metrics[self.so][1],
            self.best_solution.metrics[self.fo][0]
            * self.best_solution.metrics[self.fo][1],
            color="green",
            marker="*",
            linestyle="None",
            label="Chosen Solution",
        )
        plt.xlabel(f"{self.so}")
        plt.ylabel(f"{self.fo}")
        plt.title(f"{self.fo} x {self.so}")
        plt.legend()

    def solve(
        self, mutation_rate: float = 0.05, generations: int = 0
    ) -> Individual:
        """Solves the optimization problem using a genetic algorithm.

        Initializes the population, evaluates the fitness of individuals, and iteratively
        evolves the population over a specified number of generations. At each generation,
        individuals are selected, crossed over, and mutated to generate a new population.
        Pareto dominance is used to maintain a Pareto frontier of non-dominated solutions.
        Finally, the Pareto frontier is visualized and the best solution is chosen by
        Lexicographic Approach.

        Args:
            mutation_rate (float): The mutation rate for the genetic algorithm.
            generations (int): The number of generations to evolve the population.

        Returns:
            Individual: The best solution found by the genetic algorithm.
        """
        self.init_population()

        print("Generation 0")

        for i in tqdm(range(len(self.population))):
            self.population[i]._evaluate(
                self.X_train, self.X_test, self.y_train, self.y_test
            )

        for i in range(generations):
            for individual in self.population:
                
                # The pareto frontier is calculated by finding if a solution is dominated by at least one other solution
                # The logic is given by: not exists y | f1(x) > f1(y) and f2(x) > f2(y)
                frontier = not any(
                    (
                        individual.metrics[self.fo][0]
                        * individual.metrics[self.fo][1]
                        > other_individual.metrics[self.fo][0]
                        * other_individual.metrics[self.fo][1]
                        and individual.metrics[self.so][0]
                        * individual.metrics[self.so][1]
                        > other_individual.metrics[self.so][0]
                        * other_individual.metrics[self.so][1]
                    )
                    for other_individual in self.population
                )

                if frontier:
                    self.pareto.append(individual)
            self.frontier.append(self.pareto)
            new_population = []

            # if the pareto has size 1, one of the parents is that solution
            if len(self.pareto) == 1:
                for _ in range(0, self.population_size, 2):
                    parent1 = self.select_parent(len(self.pareto))
                    parent2 = self.select_parent(self.population_size)

                    children = self.pareto[parent1].crossover(
                        self.population[parent2]
                    )

                    new_population.append(children[0].mutation(mutation_rate))
                    new_population.append(children[1].mutation(mutation_rate))
                    self.best_solution = self.pareto[0]

            # If not, the parents are chosen by the pareto solutions
            else:
                for _ in range(0, self.population_size, 2):
                    parent1 = self.select_parent(len(self.pareto))
                    parent2 = self.select_parent(len(self.pareto))

                    children = self.pareto[parent1].crossover(
                        self.pareto[parent2]
                    )

                    new_population.append(children[0].mutation(mutation_rate))
                    new_population.append(children[1].mutation(mutation_rate))
                    self.best_solution = self.best_individual(self.pareto)
            self.visualize_generation(self.best_solution)
            print(f"Generation {i + 1}")

            self.population = list(new_population)

            for i in tqdm(range(len(self.population))):
                self.population[i]._evaluate(
                    self.X_train, self.X_test, self.y_train, self.y_test
                )

        self.frontier = [item for row in self.frontier for item in row]

        # The final pareto frontier is calculated over all the pareto frontier solutions obtained before
        self.pareto_frontier = []
        for individual in self.frontier:
            frontier = not any(
                (
                    individual.metrics[self.fo][0]
                    * individual.metrics[self.fo][1]
                    > other_individual.metrics[self.fo][0]
                    * other_individual.metrics[self.fo][1]
                    and individual.metrics[self.so][0]
                    * individual.metrics[self.so][1]
                    > other_individual.metrics[self.so][0]
                    * other_individual.metrics[self.so][1]
                )
                for other_individual in self.frontier
            )

            if frontier:
                self.pareto_frontier.append(individual)
        self.pareto_result()

        return self.best_solution
