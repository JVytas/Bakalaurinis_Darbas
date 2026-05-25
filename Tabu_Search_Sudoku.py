"""
Tabu Search algorithm for solving 4x4 and 9x9 Sudoku puzzles.

The program keeps the original fixed Sudoku values unchanged and starts from
a row-valid candidate solution. At each iteration, it generates neighboring
solutions by swapping two non-fixed values in a randomly selected row.

A tabu list stores recently performed reverse moves, which helps the search
avoid cycling back to recently visited states. The aspiration criterion allows
a tabu move if it improves the best solution found so far.

The experiment compares different tabu list sizes and records the success rate,
the average number of iterations and the best fitness value found.
"""

import random
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import trange


# -----------------------------
# Experiment configuration
# -----------------------------

RANDOM_SEED = 202501181854
random.seed(RANDOM_SEED)

REPEATS_TS = 10

TABU_LIST_SIZE_4X4 = [5, 10]
MAX_ITERATIONS_TS_4X4 = 5000

TABU_LIST_SIZE_9X9 = [10, 20]
MAX_ITERATIONS_TS_9X9 = 1000

TABU_LIST_SIZE_4X4_EXTENDED = [1, 2, 3, 5, 7, 10]
MAX_ITERATIONS_TS_4X4_EXTENDED = 1000

TABU_LIST_SIZE_9X9_EXTENDED = [5, 10, 20, 30, 40, 50]
MAX_ITERATIONS_TS_9X9_EXTENDED = 1000


# -----------------------------
# Sudoku boards
# -----------------------------

SUDOKU_4X4 = np.array([
    [1, 2, 0, 0],
    [3, 4, 0, 0],
    [0, 0, 1, 2],
    [0, 0, 3, 4]
])

SUDOKU_9X9 = np.array([
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


# -----------------------------
# Helper functions
# -----------------------------

def get_fixed_positions(board, n):
    """
    Identify fixed positions in each row of the original Sudoku board.

    Args:
        board (np.array): Initial Sudoku board.
        n (int): Board size.

    Returns:
        list[list[int]]: For each row, a list of fixed column indices.
    """
    fixed_pos = []
    for row_idx in range(n):
        row_fixed_pos = []
        for col_idx in range(n):
            if board[row_idx][col_idx] != 0:
                row_fixed_pos.append(col_idx)
        fixed_pos.append(row_fixed_pos)
    return fixed_pos


def create_individual(board, fixed_pos, n):
    """
    Create an initial candidate solution.

    Empty cells in each row are filled with the missing numbers from 1 to n.
    This keeps every row valid and preserves the fixed Sudoku values.
    """
    individual = deepcopy(board)

    for row_idx in range(n):
        missing_numbers = list(range(1, n + 1))

        for col_idx in fixed_pos[row_idx]:
            if board[row_idx][col_idx] in missing_numbers:
                missing_numbers.remove(board[row_idx][col_idx])

        random.shuffle(missing_numbers)

        for col_idx in range(n):
            if board[row_idx][col_idx] == 0:
                individual[row_idx][col_idx] = missing_numbers.pop(0)

    return individual


def fitness(individual, n):
    """
    Count Sudoku conflicts in rows, columns and blocks.

    Lower fitness is better. A fitness value of 0 means that the board is a
    valid Sudoku solution.
    """
    violations = 0
    block_size = int(np.sqrt(n))

    # Row and column violations.
    for i in range(n):
        violations += n - len(set(individual[i, :]))
        violations += n - len(set(individual[:, i]))

    # Block violations.
    for row_start in range(0, n, block_size):
        for col_start in range(0, n, block_size):
            block = individual[
                row_start:row_start + block_size,
                col_start:col_start + block_size
            ].flatten()
            violations += n - len(set(block))

    return violations


def generate_tabu_neighbor(individual, fixed_pos, n):
    """
    Generate one neighboring solution.

    The neighbor is created by selecting a random row and swapping two
    non-fixed values in that row. The performed move and its reverse are
    returned so that the tabu list can be updated.
    """
    new_individual = deepcopy(individual)
    row_idx = random.randint(0, n - 1)

    mutable_indices = [
        col_idx for col_idx in range(n)
        if col_idx not in fixed_pos[row_idx]
    ]

    if len(mutable_indices) < 2:
        return new_individual, None, None

    col_1, col_2 = random.sample(mutable_indices, 2)

    old_value_1 = new_individual[row_idx][col_1]
    old_value_2 = new_individual[row_idx][col_2]

    new_individual[row_idx][col_1], new_individual[row_idx][col_2] = (
        new_individual[row_idx][col_2],
        new_individual[row_idx][col_1]
    )

    move_1 = (row_idx, col_1, old_value_1, old_value_2)
    move_2 = (row_idx, col_2, old_value_2, old_value_1)

    return new_individual, move_1, move_2


# -----------------------------
# Tabu Search algorithm
# -----------------------------

def tabu_search(board, n, iterations, tabu_list_size):
    """
    Run Tabu Search on a Sudoku board.

    Args:
        board (np.array): Initial Sudoku board.
        n (int): Board size.
        iterations (int): Maximum number of iterations.
        tabu_list_size (int): Maximum number of tabu moves stored.

    Returns:
        tuple: (solved, iterations_taken, best_solution)
    """
    fixed_pos = get_fixed_positions(board, n)

    current_solution = create_individual(board, fixed_pos, n)
    current_fitness = fitness(current_solution, n)

    best_solution = deepcopy(current_solution)
    best_fitness = current_fitness

    # Stores reverse moves so that recently undone moves are temporarily forbidden.
    tabu_list = []

    for iteration in trange(
        iterations,
        desc=f"TS N={n}, TabuSize={tabu_list_size}",
        leave=False
    ):
        if best_fitness == 0:
            return True, iteration, best_solution

        # Generate a set of candidate neighboring solutions.
        neighbors = []
        for _ in range(n * n):
            neighbor, move_1, move_2 = generate_tabu_neighbor(
                current_solution,
                fixed_pos,
                n
            )
            if move_1 is not None and move_2 is not None:
                neighbors.append((neighbor, move_1, move_2))

        if not neighbors:
            continue

        best_neighbor = None
        best_neighbor_fitness = float("inf")
        best_neighbor_move = None

        for neighbor_solution, move_1, move_2 in neighbors:
            neighbor_fitness = fitness(neighbor_solution, n)

            reverse_move_1 = ((move_1[0], move_1[1]), move_1[2], move_1[3])
            reverse_move_2 = ((move_2[0], move_2[1]), move_2[2], move_2[3])

            is_tabu_move = (
                reverse_move_1 in tabu_list or reverse_move_2 in tabu_list
            )

            # Aspiration criterion: accept a tabu move if it improves the
            # best solution found so far.
            if is_tabu_move and neighbor_fitness < best_fitness:
                best_neighbor = neighbor_solution
                best_neighbor_fitness = neighbor_fitness
                best_neighbor_move = (move_1, move_2)
                break

            # Otherwise, select the best non-tabu neighbor.
            if not is_tabu_move and neighbor_fitness < best_neighbor_fitness:
                best_neighbor = neighbor_solution
                best_neighbor_fitness = neighbor_fitness
                best_neighbor_move = (move_1, move_2)

        if best_neighbor is None:
            # If all generated moves are tabu and aspiration is not satisfied,
            # restart from a newly generated row-valid individual.
            candidate_neighbor = create_individual(board, fixed_pos, n)
            candidate_fitness = fitness(candidate_neighbor, n)

            if candidate_fitness < current_fitness:
                current_solution = candidate_neighbor
                current_fitness = candidate_fitness

            continue

        current_solution = best_neighbor
        current_fitness = best_neighbor_fitness

        # Add reverse moves to the tabu list.
        if best_neighbor_move is not None:
            move_to_tabu_1 = (
                (best_neighbor_move[0][0], best_neighbor_move[0][1]),
                best_neighbor_move[0][3],
                best_neighbor_move[0][2]
            )
            move_to_tabu_2 = (
                (best_neighbor_move[1][0], best_neighbor_move[1][1]),
                best_neighbor_move[1][3],
                best_neighbor_move[1][2]
            )

            tabu_list.append(move_to_tabu_1)
            tabu_list.append(move_to_tabu_2)

            # Two reverse moves are added after each swap, so two are removed
            # when the list exceeds its limit.
            if len(tabu_list) > tabu_list_size:
                tabu_list.pop(0)
                tabu_list.pop(0)

        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = deepcopy(current_solution)

    return False, iterations, best_solution


# -----------------------------
# Experiment runner
# -----------------------------

def run_experiment_ts(sudoku_board, n, tabu_list_sizes, repeats=1, max_iterations=1000):
    """
    Run Tabu Search experiments for several tabu list sizes.

    Returns:
        pd.DataFrame: Summary table with success rate, average iterations and
                      best fitness value.
    """
    results = []

    for tabu_size in tabu_list_sizes:
        success_count = 0
        total_iterations = 0
        best_overall_board = None
        best_overall_fitness = float("inf")

        for repeat_idx in trange(
            repeats,
            desc=f"TS N={n}, TabuSize={tabu_size}",
            leave=False
        ):
            solved, iterations_taken, final_board = tabu_search(
                board=sudoku_board,
                n=n,
                iterations=max_iterations,
                tabu_list_size=tabu_size
            )

            final_fitness = fitness(final_board, n)

            if solved:
                success_count += 1
                print(
                    f"Solved N={n} with TabuSize={tabu_size} "
                    f"in {iterations_taken} iterations "
                    f"(Repeat {repeat_idx + 1}/{repeats})"
                )
                print(final_board)

            if final_fitness < best_overall_fitness:
                best_overall_fitness = final_fitness
                best_overall_board = final_board

            total_iterations += iterations_taken

        results.append({
            "Lentelės dydis": f"{n}x{n}",
            "Tabu List Size": tabu_size,
            "Sėkmės dažnis": f"{success_count}/{repeats}",
            "Vid. iteracijų": round(total_iterations / repeats, 2),
            "Geriausias fitness": best_overall_fitness
        })

        if success_count == 0 and best_overall_board is not None:
            print(
                f"For N={n}, TabuSize={tabu_size}: no solution found. "
                f"Best fitness: {best_overall_fitness}"
            )
            print(best_overall_board)

    return pd.DataFrame(results)


def run_all_experiments():
    """
    Run all Tabu Search experiments and save results to CSV files.
    """
    print("\n--- Running Tabu Search Experiments ---\n")

    df_ts_4x4 = run_experiment_ts(
        SUDOKU_4X4,
        4,
        TABU_LIST_SIZE_4X4,
        REPEATS_TS,
        MAX_ITERATIONS_TS_4X4
    )

    df_ts_9x9 = run_experiment_ts(
        SUDOKU_9X9,
        9,
        TABU_LIST_SIZE_9X9,
        REPEATS_TS,
        MAX_ITERATIONS_TS_9X9
    )

    df_ts_4x4_extended = run_experiment_ts(
        SUDOKU_4X4,
        4,
        TABU_LIST_SIZE_4X4_EXTENDED,
        REPEATS_TS,
        MAX_ITERATIONS_TS_4X4_EXTENDED
    )

    df_ts_9x9_extended = run_experiment_ts(
        SUDOKU_9X9,
        9,
        TABU_LIST_SIZE_9X9_EXTENDED,
        REPEATS_TS,
        MAX_ITERATIONS_TS_9X9_EXTENDED
    )

    print("\n4x4 results:")
    print(df_ts_4x4)

    print("\n9x9 results:")
    print(df_ts_9x9)

    print("\n4x4 extended results:")
    print(df_ts_4x4_extended)

    print("\n9x9 extended results:")
    print(df_ts_9x9_extended)

    df_ts_4x4.to_csv("tabu_4x4_results.csv", index=False)
    df_ts_9x9.to_csv("tabu_9x9_results.csv", index=False)
    df_ts_4x4_extended.to_csv("tabu_4x4_extended_results.csv", index=False)
    df_ts_9x9_extended.to_csv("tabu_9x9_extended_results.csv", index=False)

    return df_ts_4x4, df_ts_9x9, df_ts_4x4_extended, df_ts_9x9_extended


if __name__ == "__main__":
    run_all_experiments()
