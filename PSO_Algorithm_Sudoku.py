"""
Particle Swarm Optimization (PSO) for solving 4x4 and 9x9 Sudoku puzzles.

This implementation adapts the PSO idea to a discrete Sudoku problem.
In classical PSO, particles move through a continuous search space using
positions and velocities. In this simplified discrete version, a particle is
represented as a filled Sudoku board, and the "movement" of a particle is
approximated by swapping or changing values in non-fixed cells.

The experiment compares different swarm sizes and PSO parameters:
- c1: cognitive coefficient,
- c2: social coefficient,
- w: inertia weight.

For each parameter combination, the algorithm is repeated several times.
The output records the success rate and the average number of iterations.
"""

import random
import math
import pandas as pd
from tqdm import trange


# -----------------------------
# Experiment configuration
# -----------------------------

RANDOM_SEED = 202501181854
random.seed(RANDOM_SEED)

REPEATS_PSO = 10

# Parameters for 4x4 Sudoku experiments
SWARM_SIZES_4X4 = [10, 20]
C1_VALUES_4X4 = [0.5, 1.0]
C2_VALUES_4X4 = [1.5, 2.0]
W_VALUES_4X4 = [0.7, 0.9]
MAX_ITERATIONS_4X4 = 1000

# Parameters for 9x9 Sudoku experiments
SWARM_SIZES_9X9 = [50, 100]
C1_VALUES_9X9 = [0.5, 1.0]
C2_VALUES_9X9 = [1.5, 2.0]
W_VALUES_9X9 = [0.7, 0.9]
MAX_ITERATIONS_9X9 = 1000

RESULTS_4X4_FILE = "pso_sudoku_4x4_results.csv"
RESULTS_9X9_FILE = "pso_sudoku_9x9_results.csv"


# -----------------------------
# Sudoku boards
# -----------------------------

# 0 represents an empty cell.
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


# -----------------------------
# Fitness function
# -----------------------------

def fitness(board, n):
    """
    Calculate the number of Sudoku rule violations.

    A lower value means a better solution. A fitness value of 0 means that
    the board satisfies all row, column and block constraints.
    """
    violations = 0

    # Row violations
    for row in range(n):
        seen = set()
        for col in range(n):
            value = board[row][col]
            if value != 0:
                if value in seen:
                    violations += 1
                seen.add(value)

    # Column violations
    for col in range(n):
        seen = set()
        for row in range(n):
            value = board[row][col]
            if value != 0:
                if value in seen:
                    violations += 1
                seen.add(value)

    # Block violations
    block_size = int(math.sqrt(n))
    for block_row_start in range(0, n, block_size):
        for block_col_start in range(0, n, block_size):
            seen = set()
            for row in range(block_row_start, block_row_start + block_size):
                for col in range(block_col_start, block_col_start + block_size):
                    value = board[row][col]
                    if value != 0:
                        if value in seen:
                            violations += 1
                        seen.add(value)

    return violations


# -----------------------------
# PSO algorithm
# -----------------------------

def particle_swarm_optimization(initial_board, n, iterations, swarm_size, c1, c2, w):
    """
    Run the adapted PSO algorithm for a Sudoku board.

    Each particle is a candidate Sudoku solution. Since Sudoku is a discrete
    problem, the particle update is approximated by randomly perturbing the
    values in non-fixed cells. The number of perturbations is influenced by
    the PSO parameters c1, c2 and w.

    Returns:
        solved (bool): True if a valid solution was found.
        iterations_taken (int): Number of iterations used.
        best_board (list): Best board found.
    """
    # Store fixed cells from the original puzzle. These values must not change.
    fixed_cells = [
        (row, col)
        for row in range(n)
        for col in range(n)
        if initial_board[row][col] != 0
    ]

    def initialize_board_with_random_fill(board):
        """
        Fill empty cells of a Sudoku board with random values from 1 to n.
        """
        new_board = [row[:] for row in board]
        for row in range(n):
            for col in range(n):
                if new_board[row][col] == 0:
                    new_board[row][col] = random.randint(1, n)
        return new_board

    class Particle:
        """
        A particle stores its current board and the best board it has found.
        """
        def __init__(self, board_template):
            self.board = initialize_board_with_random_fill(board_template)
            self.pbest_board = [row[:] for row in self.board]
            self.pbest_fitness = fitness(self.board, n)

    # Initialize swarm
    swarm = [Particle(initial_board) for _ in range(swarm_size)]

    # Find the initial global best solution.
    gbest_board = None
    gbest_fitness = float("inf")

    for particle in swarm:
        if particle.pbest_fitness < gbest_fitness:
            gbest_fitness = particle.pbest_fitness
            gbest_board = [row[:] for row in particle.pbest_board]

    if gbest_fitness == 0:
        return True, 0, gbest_board

    # Main optimization loop
    for iteration in range(iterations):
        for particle in swarm:
            current_board = [row[:] for row in particle.board]

            # Only non-fixed cells can be changed.
            mutable_cells = [
                (row, col)
                for row in range(n)
                for col in range(n)
                if (row, col) not in fixed_cells
            ]

            if not mutable_cells:
                continue

            # Simplified discrete "movement":
            # swap values in randomly selected mutable cells.
            num_perturbations = max(
                1,
                min(
                    len(mutable_cells) // 2,
                    int(swarm_size * (c1 + c2 + w) / (n * n)) + 1
                )
            )

            for _ in range(num_perturbations):
                if len(mutable_cells) < 2:
                    break

                first_index, second_index = random.sample(range(len(mutable_cells)), 2)
                row1, col1 = mutable_cells[first_index]
                row2, col2 = mutable_cells[second_index]

                current_board[row1][col1], current_board[row2][col2] = (
                    current_board[row2][col2],
                    current_board[row1][col1],
                )

            # Additional random change to increase exploration.
            if mutable_cells and random.random() < 0.2 * (c1 + c2):
                row, col = random.choice(mutable_cells)
                current_board[row][col] = random.randint(1, n)

            new_fitness = fitness(current_board, n)

            # Update personal best.
            if new_fitness < particle.pbest_fitness:
                particle.pbest_fitness = new_fitness
                particle.pbest_board = [row[:] for row in current_board]

            particle.board = [row[:] for row in current_board]

        # Update global best after all particles have moved.
        for particle in swarm:
            if particle.pbest_fitness < gbest_fitness:
                gbest_fitness = particle.pbest_fitness
                gbest_board = [row[:] for row in particle.pbest_board]

        if gbest_fitness == 0:
            return True, iteration + 1, gbest_board

    return False, iterations, gbest_board


# -----------------------------
# Experiment runner
# -----------------------------

def run_experiment_pso(
    sudoku_board,
    n,
    swarm_sizes,
    c1_values,
    c2_values,
    w_values,
    repeats=1,
    max_iterations=1000
):
    """
    Run PSO experiments for all parameter combinations.

    Returns a pandas DataFrame with success rate and average iterations.
    """
    results = []

    for swarm_size in swarm_sizes:
        for c1 in c1_values:
            for c2 in c2_values:
                for w in w_values:
                    success_count = 0
                    total_iterations = 0
                    best_overall_board = None
                    best_overall_fitness = float("inf")

                    for _ in trange(
                        repeats,
                        desc=f"PSO N={n}, Swarm={swarm_size}, C1={c1}, C2={c2}, W={w}",
                        leave=False
                    ):
                        solved, iterations_taken, final_board = particle_swarm_optimization(
                            initial_board=sudoku_board,
                            n=n,
                            iterations=max_iterations,
                            swarm_size=swarm_size,
                            c1=c1,
                            c2=c2,
                            w=w
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
                        "Swarm Size": swarm_size,
                        "C1": c1,
                        "C2": c2,
                        "W": w,
                        "Sėkmės dažnis": f"{success_count}/{repeats}",
                        "Vid. iteracijų": round(total_iterations / repeats, 2),
                        "Geriausias fitness": best_overall_fitness,
                    })

                    if success_count == 0 and best_overall_board is not None:
                        print(
                            f"No solution found for N={n}, Swarm={swarm_size}, "
                            f"C1={c1}, C2={c2}, W={w}. "
                            f"Best fitness: {best_overall_fitness}"
                        )

    return pd.DataFrame(results)


def run_all_experiments():
    """
    Run experiments for both 4x4 and 9x9 Sudoku boards.
    """
    print("\n--- Running Particle Swarm Optimization Experiments ---\n")

    df_pso_4x4 = run_experiment_pso(
        SUDOKU_4X4,
        n=4,
        swarm_sizes=SWARM_SIZES_4X4,
        c1_values=C1_VALUES_4X4,
        c2_values=C2_VALUES_4X4,
        w_values=W_VALUES_4X4,
        repeats=REPEATS_PSO,
        max_iterations=MAX_ITERATIONS_4X4
    )

    df_pso_9x9 = run_experiment_pso(
        SUDOKU_9X9,
        n=9,
        swarm_sizes=SWARM_SIZES_9X9,
        c1_values=C1_VALUES_9X9,
        c2_values=C2_VALUES_9X9,
        w_values=W_VALUES_9X9,
        repeats=REPEATS_PSO,
        max_iterations=MAX_ITERATIONS_9X9
    )

    print("\nPSO 4x4 results:")
    print(df_pso_4x4)

    print("\nPSO 9x9 results:")
    print(df_pso_9x9)

    df_pso_4x4.to_csv(RESULTS_4X4_FILE, index=False)
    df_pso_9x9.to_csv(RESULTS_9X9_FILE, index=False)

    print(f"\nResults saved to {RESULTS_4X4_FILE} and {RESULTS_9X9_FILE}")

    return df_pso_4x4, df_pso_9x9


if __name__ == "__main__":
    run_all_experiments()
