"""
Memetic algorithm for DNA motif search.

This program generates random DNA sequences, inserts a known target motif
("AAAAAAAAAA") into each sequence, and then tries to recover this motif
using a memetic algorithm.

The algorithm combines:
- a population of candidate motifs,
- crossover,
- mutation,
- local search.

The maximum possible fitness score is:

    number_of_sequences * motif_length = 100 * 10 = 1000

Experiments are repeated for different mutation rates in order to evaluate
how mutation affects the average number of generations needed to find
the target motif.
"""

import random
import csv


# -----------------------------
# Experiment configuration
# -----------------------------

RANDOM_SEED = 202501181854

NUM_SEQUENCES = 100
SEQUENCE_LENGTH = 100

TARGET_MOTIF = "AAAAAAAAAA"
MOTIF_LENGTH = len(TARGET_MOTIF)

POPULATION_SIZE = 100
GENERATIONS = 100
NUM_RUNS = 100

MUTATION_RATES = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

RESULTS_FILE = "dna_motif_results.csv"


# -----------------------------
# Data generation
# -----------------------------

def generate_random_sequences(num_sequences, sequence_length):
    """
    Generate random DNA sequences consisting of A, C, G and T.
    """
    return [
        "".join(random.choice("ACGT") for _ in range(sequence_length))
        for _ in range(num_sequences)
    ]


def incorporate_motif(sequence, motif):
    """
    Insert the target motif into a random position of one DNA sequence.
    """
    position = random.randint(0, len(sequence) - len(motif))
    return sequence[:position] + motif + sequence[position + len(motif):]


def incorporate_for_all(sequences, motif):
    """
    Insert the target motif into every generated DNA sequence.
    """
    return [incorporate_motif(sequence, motif) for sequence in sequences]


# -----------------------------
# Fitness and population
# -----------------------------

def fitness(motif, sequences):
    """
    Evaluate motif quality.

    For each DNA sequence, the function finds the best local match
    for the candidate motif. The final fitness score is the sum of
    the best match scores across all sequences.

    A perfect score is:
        len(sequences) * len(motif)
    """
    motif_length = len(motif)
    total_score = 0

    for sequence in sequences:
        best_score_for_sequence = 0

        for i in range(len(sequence) - motif_length + 1):
            subsequence = sequence[i:i + motif_length]
            match_score = sum(
                1 for a, b in zip(motif, subsequence) if a == b
            )

            if match_score > best_score_for_sequence:
                best_score_for_sequence = match_score

        total_score += best_score_for_sequence

    return total_score


def generate_initial_population(pop_size, motif_length):
    """
    Generate an initial population of random candidate motifs.
    """
    return [
        "".join(random.choice("ACGT") for _ in range(motif_length))
        for _ in range(pop_size)
    ]


# -----------------------------
# Genetic operators
# -----------------------------

def crossover(parent1, parent2):
    """
    Perform one-point crossover between two parent motifs.
    """
    cut = random.randint(1, len(parent1) - 1)
    return parent1[:cut] + parent2[cut:]


def mutate(motif, mutation_rate):
    """
    Mutate a motif.

    Each nucleotide has a probability equal to mutation_rate
    to be replaced by a random nucleotide.
    """
    return "".join(
        random.choice("ACGT") if random.random() < mutation_rate else nucleotide
        for nucleotide in motif
    )


def local_search(motif, sequences):
    """
    Improve a motif using local search.

    Each position of the motif is tested with all possible DNA bases.
    If a change improves the fitness score, it is kept.
    """
    best_motif = motif
    best_score = fitness(motif, sequences)

    for i in range(len(motif)):
        for base in "ACGT":
            candidate = motif[:i] + base + motif[i + 1:]
            candidate_score = fitness(candidate, sequences)

            if candidate_score > best_score:
                best_motif = candidate
                best_score = candidate_score

    return best_motif


# -----------------------------
# Memetic algorithm
# -----------------------------

def memetic_algorithm(sequences, motif_length, pop_size, generations, mutation_rate):
    """
    Run the memetic algorithm and return the number of generations needed
    to find the target motif.

    If a perfect motif is not found, the function returns the maximum
    number of generations.
    """
    population = generate_initial_population(pop_size, motif_length)
    max_score = len(sequences) * motif_length

    for generation in range(1, generations + 1):
        # Improve all motifs by local search.
        population = [local_search(motif, sequences) for motif in population]

        # Sort population by fitness. Higher score means better motif.
        population.sort(key=lambda motif: fitness(motif, sequences), reverse=True)

        best_motif = population[0]
        best_score = fitness(best_motif, sequences)

        # Stop when a perfect motif is found.
        if best_score == max_score:
            return generation

        # Create next generation.
        next_population = [best_motif]  # elitism: keep the best motif

        while len(next_population) < pop_size:
            # Select parents from the better half of the population.
            parent_pool = population[:max(2, pop_size // 2)]
            parent1, parent2 = random.sample(parent_pool, 2)

            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            child = local_search(child, sequences)

            next_population.append(child)

        population = next_population

    return generations


# -----------------------------
# Experiment runner
# -----------------------------

def run_experiments():
    """
    Run experiments for different mutation rates and save results to CSV.
    """
    random.seed(RANDOM_SEED)

    base_sequences = generate_random_sequences(NUM_SEQUENCES, SEQUENCE_LENGTH)
    sequences_with_motif = incorporate_for_all(base_sequences, TARGET_MOTIF)

    results = []

    for mutation_rate in MUTATION_RATES:
        total_generations = 0

        for _ in range(NUM_RUNS):
            generations_taken = memetic_algorithm(
                sequences=sequences_with_motif,
                motif_length=MOTIF_LENGTH,
                pop_size=POPULATION_SIZE,
                generations=GENERATIONS,
                mutation_rate=mutation_rate
            )

            total_generations += generations_taken

        average_generations = total_generations / NUM_RUNS

        results.append({
            "mutation_rate": mutation_rate,
            "average_generations": round(average_generations, 2)
        })

        print(
            f"Mutation rate: {mutation_rate:.2f}, "
            f"Average generations: {average_generations:.2f}"
        )

    save_results(results)


def save_results(results):
    """
    Save experiment results to a CSV file.
    """
    with open(RESULTS_FILE, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["mutation_rate", "average_generations"]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_experiments()
