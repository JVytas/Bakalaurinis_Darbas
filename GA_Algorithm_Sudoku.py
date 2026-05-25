"""
Genetic algorithm for solving 4x4 and 9x9 Sudoku puzzles.

The program keeps the original fixed Sudoku values unchanged and generates
candidate solutions by filling the remaining cells. Each row is initialized
as a valid permutation, so the fitness function mainly evaluates column and
block conflicts.

The experiment compares different population sizes and mutation rates.
For each parameter combination, the algorithm is repeated several times and
the success rate and average number of generations are recorded.
"""

import numpy as np
import random
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from IPython.display import display

# Fixed random seed for reproducibility.
RANDOM_SEED = 202501181854
random.seed(RANDOM_SEED)

# Sudoku board (9x9). Zeros represent empty cells.
initial_board_9x9 = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
])

# Sudoku board (4x4). Zeros represent empty cells.
initial_board_4x4 = np.array([
    [1, 2, 0, 0],
    [3, 4, 0, 0],
    [0, 0, 1, 2],
    [0, 0, 3, 4]
])


def get_fixed_positions(board):
    """
    Identify the positions of pre-filled numbers on the Sudoku board.

    These positions represent the fixed cells of the puzzle and cannot be
    changed by crossover or mutation.

    Args:
        board (np.array): Sudoku board of size N x N.

    Returns:
        list of lists: For each row, a list of column indices containing
        fixed values.
    """
    N = board.shape[0]
    return [[i for i in range(N) if row[i] != 0] for row in board]


def create_individual(board, fixed_pos):
    """
    Generate one candidate Sudoku solution.

    Empty cells in each row are filled with the missing numbers from 1 to N.
    This keeps each row valid at initialization, while fixed cells remain
    unchanged.

    Args:
        board (np.array): Initial Sudoku board.
        fixed_pos (list of lists): Fixed cell positions for each row.

    Returns:
        np.array: Candidate Sudoku solution.
    """
    N = board.shape[0]
    individual = deepcopy(board)

    for i in range(N):
        missing = [n for n in range(1, N + 1) if n not in board[i]]
        random.shuffle(missing)
        idx = 0

        for j in range(N):
            if j not in fixed_pos[i]:
                individual[i][j] = missing[idx]
                idx += 1

    return individual


def fitness(individual):
    """
    Evaluate the quality of a candidate Sudoku solution.

    Lower fitness means a better solution. A value of 0 means that the board
    satisfies Sudoku constraints. Rows are already kept valid by the
    representation, so the function mainly penalizes column and block conflicts.

    Args:
        individual (np.array): Candidate Sudoku solution.

    Returns:
        int: Number of conflicts in columns and blocks.
    """
    N = individual.shape[0]
    block_size = int(np.sqrt(N))
    score = 0

    # Column conflicts.
    for col in range(N):
        score += N - len(set(individual[:, col]))

    # Block conflicts.
    for i in range(block_size):
        for j in range(block_size):
            square = individual[
                i * block_size:(i + 1) * block_size,
                j * block_size:(j + 1) * block_size
            ].flatten()
            score += N - len(set(square))

    return score


def mutate(individual, fixed_pos):
    """
    Mutate an individual by swapping two mutable cells in one random row.

    The mutation preserves the row-based representation and does not modify
    the original fixed Sudoku values.

    Args:
        individual (np.array): Candidate solution to mutate.
        fixed_pos (list of lists): Fixed cell positions for each row.

    Returns:
        np.array: Mutated candidate solution.
    """
    N = individual.shape[0]
    i = random.randint(0, N - 1)
    row = individual[i]
    mutable_indices = [j for j in range(N) if j not in fixed_pos[i]]

    if len(mutable_indices) >= 2:
        a, b = random.sample(mutable_indices, 2)
        row[a], row[b] = row[b], row[a]

    return individual


def crossover(parent1, parent2):
    """
    Create a child solution from two parent solutions.

    For each row, the child inherits the full row from either parent with equal
    probability. This keeps rows internally valid if the parents' rows are valid.

    Args:
        parent1 (np.array): First parent solution.
        parent2 (np.array): Second parent solution.

    Returns:
        np.array: Child solution.
    """
    N = parent1.shape[0]
    child = np.zeros((N, N), dtype=int)

    for i in range(N):
        child[i] = parent1[i] if random.random() < 0.5 else parent2[i]

    return child


def genetic_algorithm(board, generations=1000, pop_size=1000, mutation_rate=0.1):
    """
    Run the genetic algorithm for a Sudoku board.

    Args:
        board (np.array): Initial Sudoku board.
        generations (int): Maximum number of generations.
        pop_size (int): Population size.
        mutation_rate (float): Probability of mutating a child.

    Returns:
        tuple: (generation_count, success, solution). If no solution is found,
        success is False and solution is None.
    """
    fixed_pos = get_fixed_positions(board)
    population = [create_individual(board, fixed_pos) for _ in range(pop_size)]

    for generation in range(generations):
        population = sorted(population, key=fitness)

        if fitness(population[0]) == 0:
            return generation, True, population[0]

        # Elitism: keep the best 2% of the current population.
        new_population = population[:int(pop_size * 0.02)]

        while len(new_population) < pop_size:
            # Select parents from the top 10% of the population.
            sample_pool_size = max(2, int(pop_size * 0.1))
            p1, p2 = random.sample(population[:sample_pool_size], 2)

            child = crossover(p1, p2)

            if random.random() < mutation_rate:
                child = mutate(child, fixed_pos)

            new_population.append(child)

        population = new_population

    return generations, False, None


# Experiment configuration.
board_configs = {
    "9x9": initial_board_9x9,
    "4x4": initial_board_4x4,
}

# Population sizes are chosen separately for 9x9 and 4x4 Sudoku boards.
pop_sizes_by_board = {
    "9x9": [100, 200, 300],
    "4x4": [10, 20, 30],
}

mutation_rates = [0.05, 0.1]
num_trials = 10  # Number of independent runs for each parameter combination.
results = []

for board_size_str, initial_board in board_configs.items():
    print(f"\nRunning experiments for {board_size_str} board...")
    current_pop_sizes = pop_sizes_by_board[board_size_str]

    for pop in current_pop_sizes:
        for mut in mutation_rates:
            successes = 0
            total_generations = 0

            for _ in tqdm(range(num_trials), desc=f"P={pop}, M={mut} ({board_size_str})"):
                gens, success, solution = genetic_algorithm(
                    initial_board,
                    generations=1000,
                    pop_size=pop,
                    mutation_rate=mut,
                )

                if success:
                    successes += 1
                    total_generations += gens
                else:
                    # If the algorithm fails, count it as using all allowed generations.
                    total_generations += 1000

            avg_gen = round(total_generations / num_trials, 1)
            results.append({
                "Lentelės dydis": board_size_str,
                "Populiacija": pop,
                "Mutacija": mut,
                "Sėkmės dažnis": f"{successes}/{num_trials}",
                "Vid. generacijų": avg_gen,
                "Paleidimai": num_trials,
            })

# Convert results to a table and save them.
df = pd.DataFrame(results)
display(df)
df.to_csv("sudoku_rezultatai.csv", index=False)
