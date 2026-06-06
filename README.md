# Bachelor Thesis - “Genetic Algorithms and Their Applications in Natural and Exact Sciences”

This repository contains the code and experiments for my bachelor thesis at Vilnius University.

## Overview

The project implements and compares several metaheuristic optimization algorithms:

- Genetic Algorithm (GA)
- Memetic Algorithm (MA)
- Ant Colony Optimization (ACO)
- Particle Swarm Optimization (PSO)
- Tabu Search

The algorithms were applied to:

- DNA motif search
- Arithmetic expression optimization
- 4x4 and 9x9 Sudoku solving

The experiments involved parameter tuning, result comparison, convergence analysis and visualization.

## Technologies

Python, NumPy, pandas, tqdm, matplotlib, Jupyter Notebook / Google Colab, Git

## Files

- `MA_program.py` – memetic algorithm for DNA motif search.
- `MA_program.ipynb` – notebook version of the MA DNA motif experiment.
- `GA_program.py` – genetic algorithm for arithmetic expression optimization.
- `GA_program.ipynb` – notebook version of the GA arithmetic expression experiment.
- `GA_Algorithm_Sudoku.py` – genetic algorithm for 4x4 and 9x9 Sudoku.
- `GA_Algorithm_Sudoku.ipynb` – notebook version of the GA Sudoku experiment.
- `ACO_algorithm_Sudoku.py` – ant colony optimization for Sudoku.
- `ACO_algorithm_Sudoku.ipynb` – notebook version of the ACO Sudoku experiment.
- `PSO_Algorithm_Sudoku.py` – particle swarm optimization for Sudoku.
- `PSO_Algorithm_Sudoku.ipynb` – notebook version of the PSO Sudoku experiment.
- `Tabu_Search_Sudoku.py` – tabu search for Sudoku.
- `Tabu_Search_Sudoku.ipynb` – notebook version of the Tabu Search Sudoku experiment.

## Requirements

The programs use Python and common libraries such as NumPy, pandas, tqdm and matplotlib. Some experiments are also available as Jupyter/Google Colab notebooks.

## Notes

The Python script versions are included to make the experiments easier to review and reproduce. The notebook versions correspond to the implementations referenced in the thesis appendix.
