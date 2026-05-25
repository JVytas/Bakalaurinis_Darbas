"""
Ant Colony Optimization (ACO) algorithm for solving 4x4 and 9x9 Sudoku puzzles.

The program uses an ant colony approach where each ant constructs a candidate
Sudoku solution by filling the empty cells of the puzzle. Pheromone values are
stored for possible number placements in each cell. Better solutions reinforce
the choices that led to them, while pheromone evaporation prevents the search
from relying too strongly on earlier paths.

The experiment compares different numbers of ants, pheromone evaporation rates
and pheromone deposition constants (Q). For each parameter combination, the
algorithm is repeated several times and the success rate and average number of
iterations are recorded.
"""

import random
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import trange


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = 202501181854
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Sudoku boards
# ---------------------------------------------------------------------------

SUDOKU_4X4 = [
    [1, 2, 0, 0],
    [3, 4, 0, 0],
    [0, 0, 1, 2],
    [0, 0, 3, 4],
]

SUDOKU_9X9 = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]


# ---------------------------------------------------------------------------
# Fitness and helper functions
# ---------------------------------------------------------------------------

def fitness(board, n):
    """
    Count Sudoku conflicts in rows, columns and subgrids.

    A lower fitness value means a better solution. A fitness value of 0 means
    that there are no repeated values in any row, column or block.
    """
    conflicts = 0
    block_size = int(np.sqrt(n))

    for row in range(n):
        row_values = board[row, :]
        non_zero_values = row_values[row_values != 0]
        conflicts += len(non_zero_values) - len(np.unique(non_zero_values))

    for col in range(n):
        col_values = board[:, col]
        non_zero_values = col_values[col_values != 0]
        conflicts += len(non_zero_values) - len(np.unique(non_zero_values))

    for block_row in range(block_size):
        for block_col in range(block_size):
            block_values = board[
                block_row * block_size:(block_row + 1) * block_size,
                block_col * block_size:(block_col + 1) * block_size,
            ].flatten()
            non_zero_values = block_values[block_values != 0]
            conflicts += len(non_zero_values) - len(np.unique(non_zero_values))

    return conflicts


def get_fixed_positions(board, n):
    """
    Return the coordinates of cells that were already filled in the original puzzle.
    These cells are not changed by ants.
    """
    fixed_positions = []

    for row in range(n):
        for col in range(n):
            if board[row, col] != 0:
                fixed_positions.append((row, col))

    return fixed_positions


def is_valid_placement(board, row, col, num, n):
    """
    Check whether a number can be placed into a Sudoku cell without immediately
    violating row, column or block constraints.
    """
    if num in board[row, :]:
        return False

    if num in board[:, col]:
        return False

    block_size = int(np.sqrt(n))
    start_row = block_size * (row // block_size)
    start_col = block_size * (col // block_size)

    if num in board[start_row:start_row + block_size, start_col:start_col + block_size]:
        return False

    return True


# ---------------------------------------------------------------------------
# Pheromone handling
# ---------------------------------------------------------------------------

def initialize_pheromones(n, initial_pheromone=0.1):
    """
    Initialize a pheromone matrix for all possible cell-value choices.

    pheromones[row, col, value] stores the pheromone level for placing
    'value' into cell (row, col).
    """
    return np.full((n, n, n + 1), initial_pheromone, dtype=float)


def ant_construct_solution(board_template, pheromones, n):
    """
    Construct one candidate Sudoku solution.

    The ant fills empty cells in random order. For each empty cell, it selects
    a valid number probabilistically according to pheromone values.
    """
    current_board = deepcopy(board_template)
    empty_cells = [
        (row, col)
        for row in range(n)
        for col in range(n)
        if board_template[row, col] == 0
    ]

    random.shuffle(empty_cells)
    choices = []

    for row, col in empty_cells:
        possible_values = []
        probabilities = []

        for num in range(1, n + 1):
            if is_valid_placement(current_board, row, col, num, n):
                possible_values.append(num)
                probabilities.append(pheromones[row, col, num])

        if not possible_values:
            # If no valid value is available, the ant is forced to place a
            # random number. Such solutions usually receive a worse fitness
            # value and therefore receive little or no reinforcement.
            chosen_num = random.randint(1, n)
        else:
            total_probability = sum(probabilities)

            if total_probability == 0:
                chosen_num = random.choice(possible_values)
            else:
                probabilities = [p / total_probability for p in probabilities]
                chosen_num = random.choices(possible_values, weights=probabilities, k=1)[0]

        current_board[row, col] = chosen_num
        choices.append((row, col, chosen_num))

    return current_board, choices


def update_pheromones(pheromones, best_solution, initial_board, n, evaporation_rate, q):
    """
    Update pheromone trails using evaporation and deposition.

    Evaporation decreases all pheromone values. Deposition reinforces placements
    from the best solution found so far. The Q parameter controls how much
    pheromone is deposited.
    """
    pheromones *= (1 - evaporation_rate)

    best_fitness = fitness(best_solution, n)

    if best_fitness == 0:
        delta_pheromone = q
    else:
        delta_pheromone = q / (best_fitness + 1e-6)

    for row in range(n):
        for col in range(n):
            if initial_board[row][col] == 0:
                chosen_num = best_solution[row, col]

                if chosen_num != 0:
                    pheromones[row, col, chosen_num] += delta_pheromone


# ---------------------------------------------------------------------------
# Ant Colony Optimization algorithm
# ---------------------------------------------------------------------------

def ant_colony_optimization(
    board,
    n,
    iterations,
    num_ants,
    evaporation_rate,
    q,
    initial_pheromone=0.1,
):
    """
    Run ACO for a single Sudoku puzzle and parameter combination.

    Returns:
        solved (bool): True if a conflict-free solution was found.
        iterations_taken (int): Number of iterations used.
        best_overall_solution (np.array): Best board found during the run.
    """
    board_array = np.array(board)
    pheromones = initialize_pheromones(n, initial_pheromone)

    best_overall_solution = None
    best_overall_fitness = float("inf")

    for iteration in trange(
        1,
        iterations + 1,
        desc=f"ACO N={n}, Ants={num_ants}, Evap={evaporation_rate}",
        leave=False,
    ):
        current_best_ant_solution = None
        current_best_ant_fitness = float("inf")

        for _ in range(num_ants):
            ant_solution, _ = ant_construct_solution(board_array, pheromones, n)
            ant_fitness = fitness(ant_solution, n)

            if ant_fitness < current_best_ant_fitness:
                current_best_ant_fitness = ant_fitness
                current_best_ant_solution = ant_solution

        if current_best_ant_fitness < best_overall_fitness:
            best_overall_fitness = current_best_ant_fitness
            best_overall_solution = current_best_ant_solution

            if best_overall_fitness == 0:
                return True, iteration, best_overall_solution

        if best_overall_solution is not None:
            update_pheromones(
                pheromones,
                best_overall_solution,
                board,
                n,
                evaporation_rate,
                q,
            )

    return False, iterations, best_overall_solution


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment_aco(
    sudoku_board,
    n,
    num_ants_list,
    evaporation_rates,
    q_values,
    repeats=10,
    max_iterations=1000,
):
    """
    Run ACO experiments for all selected parameter combinations and return a
    pandas DataFrame with success rates and average iteration counts.
    """
    results = []

    for num_ants in num_ants_list:
        for evaporation_rate in evaporation_rates:
            for q_value in q_values:
                success_count = 0
                total_iterations = 0
                best_overall_board = None
                best_overall_fitness = float("inf")

                for _ in trange(
                    repeats,
                    desc=f"ACO N={n}, Ants={num_ants}, Evap={evaporation_rate}, Q={q_value}",
                    leave=False,
                ):
                    solved, iterations_taken, final_board = ant_colony_optimization(
                        board=sudoku_board,
                        n=n,
                        iterations=max_iterations,
                        num_ants=num_ants,
                        evaporation_rate=evaporation_rate,
                        q=q_value,
                    )

                    if solved:
                        success_count += 1

                    total_iterations += iterations_taken

                    if final_board is not None:
                        final_fitness = fitness(final_board, n)
                        if final_fitness < best_overall_fitness:
                            best_overall_fitness = final_fitness
                            best_overall_board = final_board

                results.append({
                    "Lentelės dydis": f"{n}x{n}",
                    "Skruzdžių skaičius": num_ants,
                    "Garavimo greitis": evaporation_rate,
                    "Q": q_value,
                    "Sėkmės dažnis": f"{success_count}/{repeats}",
                    "Vid. iteracijų": round(total_iterations / repeats, 2),
                    "Geriausias fitness": best_overall_fitness,
                })

                if success_count == 0 and best_overall_board is not None:
                    print(
                        f"No solution found for N={n}, ants={num_ants}, "
                        f"evaporation={evaporation_rate}, Q={q_value}. "
                        f"Best fitness: {best_overall_fitness}"
                    )

    return pd.DataFrame(results)


def main():
    """
    Run the 4x4 and 9x9 ACO Sudoku experiments.
    """
    num_ants_aco_4x4 = [10, 20]
    evaporation_rates_aco_4x4 = [0.1, 0.3]
    q_values_aco_4x4 = [0.5, 1.0]
    max_iterations_aco_4x4 = 500

    num_ants_aco_9x9 = [20, 50]
    evaporation_rates_aco_9x9 = [0.1, 0.3]
    q_values_aco_9x9 = [0.5, 1.0]
    max_iterations_aco_9x9 = 1000

    repeats_aco = 10

    print("\n--- Running Ant Colony Optimization Experiments ---\n")

    df_aco_4x4 = run_experiment_aco(
        SUDOKU_4X4,
        4,
        num_ants_aco_4x4,
        evaporation_rates_aco_4x4,
        q_values_aco_4x4,
        repeats_aco,
        max_iterations_aco_4x4,
    )

    df_aco_9x9 = run_experiment_aco(
        SUDOKU_9X9,
        9,
        num_ants_aco_9x9,
        evaporation_rates_aco_9x9,
        q_values_aco_9x9,
        repeats_aco,
        max_iterations_aco_9x9,
    )

    print("\n4x4 Sudoku ACO results:")
    print(df_aco_4x4.to_string(index=False))

    print("\n9x9 Sudoku ACO results:")
    print(df_aco_9x9.to_string(index=False))

    df_aco_4x4.to_csv("aco_sudoku_4x4_results.csv", index=False)
    df_aco_9x9.to_csv("aco_sudoku_9x9_results.csv", index=False)

    print("\nResults saved to 'aco_sudoku_4x4_results.csv' and 'aco_sudoku_9x9_results.csv'.")


if __name__ == "__main__":
    main()
