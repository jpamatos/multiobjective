# 🧬 Multiobjective Genetic Algorithm

To execute this project first create a conda environment with the command:

`conda create --name multiobjective python=3.11.5`

A new conda environment named `multiobjective` will be created. It can be activated with:

`conda activate multiobjective`

The packages needed can be installed by pip using the requirements file:

`pip install -r requirements.txt`

The genetic algorithm is executed in the `main.py` script.

<br>

---

## 📘 Project Description

This project implements a **Genetic Algorithm (GA)** for **multi-objective optimization** using the **Pareto frontier**. It is designed to evolve neural network architectures or configurations based on two performance metrics (e.g., accuracy and latency).

The final solution is selected using a **lexicographic criterion** with user-defined thresholds.

---

## 📂 Project Structure

```
.
├── src/
│   ├── genetic_algorithm.py      # Main genetic algorithm class
│   └── individual.py             # Defines the individual (model, genome, evaluation)
├── main.py                       # Entry point script
├── README.md
└── requirements.txt
```

---

## ▶️ How to Use
To run the genetic algorithm, simply execute the main.py script with your desired objectives and threshold values. Inside main.py, you can specify the objectives as follows:

```python
# User chosen objectives
first_objective = ("accuracy", 0.05)
second_objective = ("latency", 0.15)
```
You can choose any metrics available in the Individual.metrics dictionary, such as accuracy, loss, size, latency, etc.


---

## 🎯 Objectives

Each individual is evaluated according to two objective functions defined as:

- **`fo`** (First Objective): e.g., `accuracy`
- **`so`** (Second Objective): e.g., `latency`

These thresholds guide the lexicographic decision at the end.


---

## 🧬 How the Genetic Algorithm Works
1. Population Initialization
The algorithm begins by creating a population of randomly initialized Individual objects, each representing a neural network with a unique genome (hyperparameters or architecture).

2. Evaluation
Each individual is evaluated using training and testing data. The algorithm computes two objective metrics (e.g., accuracy and model size, or loss and training time) and stores them in a dictionary called metrics.

3. Pareto Dominance
After evaluation, individuals are classified into:

Pareto Frontier (Non-Dominat Solutions): Solutions for which no other individual performs better in both objectives.

Dominated Solutions: All others.

This classification is done in every generation and stored for visualization and final analysis.

4. Selection & Reproduction
Parents are selected randomly from the Pareto frontier. They reproduce using crossover, and their children are mutated with a defined mutation rate. The new generation replaces the old one.

5. Evolution
The process is repeated for a number of generations. In each generation, the Pareto frontier is updated, and a visual summary is printed showing the best current solution.

6. Final Pareto Frontier
At the end of all generations, a final Pareto frontier is computed using all previously found Pareto-optimal individuals.

---

## 🧠 Lexicographic Selection
Among all Pareto-optimal solutions, the best solution is selected using a lexicographic approach:

First Objective: The individual with the smallest (or best) value for the first objective is chosen.

Threshold Filtering: All individuals within a small tolerance of that best first-objective value are retained.

Second Objective: From that filtered group, the individual with the smallest second-objective value is selected.

Tie-breaking: If multiple individuals remain, the one with the best first objective is selected again.

This approach ensures a consistent and deterministic way of choosing a single solution from a set of trade-offs.

---

## 🛠 Requirements

- Python 3.11.5
- TensorFlow
- NumPy
- Matplotlib
- tqdm

---

## 📃 License

This is a work in progress and open for academic and educational use.

---

## ✍️ Author

Developed by Jean Matos.
