# Sovereign Optimization Solver Prototype

This project is a prototype for a sovereign mathematical optimization solver intended to reduce dependence on commercial optimization engines such as Gurobi, CPLEX and Xpress. These solvers are widely used in areas such as production planning, refinery scheduling, logistics, energy management and supply-chain optimization, but their licensing costs and closed implementations make it difficult to inspect, modify or develop solver technology around specific industrial requirements.

The goal of this project is not to build another modeling interface. The focus is on developing the **solver core itself**, starting from the mathematical foundations and gradually building toward a numerically robust engine capable of handling large-scale industrial optimization problems.

## Current Approach

The current prototype focuses on **Linear Programming (LP)** and implements a basic **primal-dual interior-point method**. The solver accepts optimization problems in the standard MPS format, preprocesses them into a standardized equality-constrained form, and then solves the resulting system using numerical linear algebra.

The preprocessing stage parses MPS files and converts `<=` and `>=` constraints by introducing slack or surplus variables. The resulting matrices are then passed to the optimization engine.

The core solver is implemented using **PyTorch**, allowing the same numerical implementation to run either on the CPU or on a CUDA-enabled GPU. At every iteration, the solver evaluates primal and dual feasibility residuals together with the complementarity/duality gap. It constructs and solves the reduced KKT system and uses backtracking to maintain positivity of the primal and dual slack variables.

A small benchmarking pipeline is also included. It can run the solver against a collection of MPS problems and record the objective value and execution time in `benchmark_results.csv`.

## What Has Been Built

The current prototype already demonstrates the main pipeline required for a solver:

**MPS input → preprocessing → standardized LP → interior-point optimization → convergence checks → benchmark results**

The repository currently contains the solver implementation, MPS benchmark datasets, a basic visualization script and generated benchmark results. The implementation also includes regularization of the KKT system and a positivity-preserving line search to make the numerical iterations more stable.

The current prototype is therefore primarily a **proof of concept for the continuous LP solving layer**, rather than the complete optimization engine described in the larger problem statement.

## Current Limitations

There are several important limitations in the current implementation.

Most importantly, the solver currently handles only a subset of LP formulations. It does not yet implement integer or binary variables, so it cannot solve MILP problems. There is also no branch-and-bound, branch-and-cut, cutting-plane or integer heuristic infrastructure yet.

The MPS preprocessing is currently simplified as well. It assumes a particular variable structure and does not yet fully handle the complete range of MPS features such as variable bounds, free variables, integer markers, quadratic objectives and all constraint/objective conventions. Quadratic programming is therefore not supported at this stage.

The current implementation also converts the problem into **dense PyTorch tensors** and explicitly constructs matrices such as `diag(X/S)`. This becomes expensive as the number of variables and constraints grows. This is currently the biggest scalability limitation and is the reason very large benchmark instances are skipped by the benchmarking script.

The interior-point implementation itself is still experimental. It does not yet have a complete infeasibility/unboundedness detection mechanism, advanced presolve, scaling strategies, crossover to a basic solution, sophisticated stopping criteria or the numerical safeguards expected from a production-grade solver. Benchmarking currently measures whether the implementation executes successfully, rather than providing a rigorous comparison of optimality against trusted reference solutions.

GPU support is also currently experimental. Although the numerical computation can be placed on CUDA, simply moving a dense optimization algorithm to a GPU does not guarantee a performance advantage. The next stages therefore need to focus on sparse matrix operations and algorithms where GPU acceleration actually provides measurable benefits.

## Proposed Development

The next stage is to move from this proof of concept toward a proper solver architecture.

The immediate priority is to replace the dense matrix representation with **sparse linear algebra**, allowing significantly larger LP instances to be processed without exhausting memory. Better scaling, presolve and numerical conditioning techniques will also be introduced to improve performance on degenerate and ill-conditioned problems.

Once the continuous LP layer becomes reliable, the solver can be extended with **MILP support through branch-and-bound**, followed by stronger components such as cutting planes, heuristics and improved node-selection strategies. This will provide the foundation for eventually targeting the large mixed-integer problems described in the original problem statement.

The longer-term architecture is intended to support LP, MILP and QP initially, while keeping the solver modular enough to eventually accommodate MIQP, NLP and MINLP.

## Running the Prototype

Clone the repository and create a Python environment:

```bash
git clone <repository-url>
cd netlib

python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:
```bash
pip install numpy torch pysmps
```

If CUDA is available and the appropriate PyTorch build is installed, the solver will automatically use the GPU. Otherwise, it falls back to the CPU.

Place MPS benchmark files inside the `feasible/` directory and run:
```bash
python solver.py
```
The solver will process the available MPS files and save the benchmark results to:
```
benchmark_results.csv
```
The current benchmark script intentionally skips very large files because the prototype still uses dense matrix operations.

## Repository Structure
```GPU-Solver/
├── feasible/             # Feasible MPS benchmark problems
├── infeasible/           # Infeasible benchmark problems
├── netlib_grb/           # Reference/benchmark data
├── solver.py             # Preprocessing and interior-point solver
├── view.py               # Benchmark/result visualization
├── benchmark_results.csv # Latest benchmark results
├── README.md
└── requirements / virtual environments
```

## Project Status

This is an early-stage research prototype. The current implementation demonstrates that an optimization problem can be parsed from a standard MPS representation, transformed into a solver-friendly form and solved using a custom interior-point implementation with CPU/GPU numerical execution.

It should not yet be considered a replacement for mature solvers such as Gurobi, CPLEX or Xpress. The purpose of the prototype is to establish the solver core and numerical approach first, then incrementally address scalability, robustness and mixed-integer optimization.

The eventual objective is to develop a transparent, extensible and independently implemented optimization engine capable of serving as a foundation for industrial optimization applications in India.
