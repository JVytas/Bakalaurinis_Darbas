"""
Genetic algorithm for arithmetic expression optimization.

The program searches for a sequence of arithmetic operators between the
fixed digits in "0123456789" so that the resulting expression evaluates
to the target value 5049.

The experiment changes the population size from 100 to 200 and repeats
each configuration 100 times. For each population size, the average number
of generations needed to find a valid expression is reported.
"""

import random


# -----------------------------
# Experiment configuration
# -----------------------------

RANDOM_SEED = 202501181854
TARGET_VALUE = 5049
NUMBERS = "0123456789"
MUTATION_RATE = 0.2
GENERATIONS = 10000
DIVERSITY_THRESHOLD = 0.8

POPULATION_SIZES = range(100, 210, 10)
NUM_REPETITIONS = 100


# -----------------------------
# Expression evaluation and fitness
# -----------------------------

def evaluate(expression):
    """
    Evaluate an arithmetic expression and return its result.

    Invalid expressions and division by zero are rejected by returning
    infinity, which makes such individuals unfit.
    """
    try:
        return eval(expression)
    except (SyntaxError, ZeroDivisionError):
        return float("inf")


def chromosome_to_expression(chromosome):
    """
    Convert a chromosome, represented as a list of arithmetic operators,
    into a full arithmetic expression using the fixed digit sequence.
    """
    expression = NUMBERS[0]
    for i, operator in enumerate(chromosome):
        expression += operator + NUMBERS[i + 1]
    return expression


def fitness(chromosome):
    """
    Calculate the fitness value of a chromosome.

    The fitness is the absolute difference between the target value and
    the value obtained from the arithmetic expression. A smaller value
    means a better solution. Fitness equal to 0 means that the target
    expression has been found.
    """
    expression = chromosome_to_expression(chromosome)
    return abs(TARGET_VALUE - evaluate(expression))


# -----------------------------
# Genetic algorithm operators
# -----------------------------

def generate_individual():
    """
    Generate a random individual.

    An individual is a sequence of arithmetic operators inserted between
    the digits in NUMBERS.
    """
    return [random.choice(["+", "*", "-", "/"]) for _ in range(len(NUMBERS) - 1)]


def crossover(parent1, parent2):
    """
    Perform one-point crossover between two parent chromosomes.
    """
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:]


def mutate(chromosome, fitness_value):
    """
    Mutate a chromosome by randomly changing one operator.

    The mutation probability is adaptive: worse individuals have a higher
    chance of mutation, while better individuals are changed less often.
    """
    mutation_rate = MUTATION_RATE * (fitness_value / TARGET_VALUE)

    if random.random() < mutation_rate:
        index = random.randint(0, len(chromosome) - 1)
        chromosome[index] = random.choice(["+", "*", "-", "/"])

    return chromosome


def calculate_diversity(population):
    """
    Calculate population diversity as the fraction of unique individuals.
    """
    unique_individuals = len(set(tuple(individual) for individual in population))
    return unique_individuals / len(population)


# -----------------------------
# Main genetic algorithm
# -----------------------------

def run_experiment(population_size, verbose=False):
    """
    Run the genetic algorithm for one population size.

    Returns:
        tuple: number of generations needed and the found expression.
               If no expression is found, returns the maximum generation
               count and an empty string.
    """
    population = [generate_individual() for _ in range(population_size)]

    for generation in range(1, GENERATIONS + 1):
        # Sort population by fitness. Lower fitness is better.
        population.sort(key=fitness)

        best_expression = chromosome_to_expression(population[0])
        best_value = evaluate(best_expression)

        if verbose:
            print(f"Generation {generation}: {best_expression} = {best_value}")

        # Check the best individual first.
        if best_value == TARGET_VALUE:
            return generation, best_expression

        # Check whether another individual in the current population already solves the task.
        for individual in population[1:]:
            expression = chromosome_to_expression(individual)
            if evaluate(expression) == TARGET_VALUE:
                return generation, expression

        # Elitism: preserve the best individual.
        next_generation = [population[0]]

        # Create the rest of the next generation using crossover and mutation.
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(population[: population_size // 2], 2)
            child = crossover(parent1, parent2)
            child_fitness = fitness(child)
            child = mutate(child, child_fitness)
            next_generation.append(child)

        # If diversity becomes too low, add new random individuals.
        diversity = calculate_diversity(next_generation)
        if diversity < DIVERSITY_THRESHOLD:
            num_random_individuals = int((1 - diversity) * population_size)
            next_generation.extend(
                [generate_individual() for _ in range(num_random_individuals)]
            )

        population = next_generation

    return GENERATIONS, ""


# -----------------------------
# Experiment runner
# -----------------------------

def run_all_experiments():
    """
    Run experiments for all population sizes and print average results.
    """
    random.seed(RANDOM_SEED)

    results = {}

    for population_size in POPULATION_SIZES:
        results[population_size] = []

        for _ in range(NUM_REPETITIONS):
            generations, _ = run_experiment(population_size)
            results[population_size].append(generations)

    for population_size, generation_counts in results.items():
        avg_generations = sum(generation_counts) / len(generation_counts)
        print(
            f"Population size: {population_size}, "
            f"Average generations to converge: {avg_generations:.2f}"
        )


if __name__ == "__main__":
    run_all_experiments()
