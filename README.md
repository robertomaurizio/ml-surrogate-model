# Fast Machine Learning Surrogate Model for Complex Physical Systems

A lightweight machine-learning surrogate model built to approximate expensive simulations in a high-dimensional physical system, enabling fast prediction, benchmarking, and tradeoff analysis.

## Problem

Many complex physical systems are governed by high-dimensional parameter spaces and require expensive numerical simulations, making exhaustive exploration computationally infeasible. This project investigates the use of machine-learning surrogate models to approximate simulation outputs and enable rapid iteration and decision-making.

## Context

The project is motivated by fusion energy research, where tokamak-based systems are used to confine ultra-hot plasma. High-fidelity simulations of these systems can require hours to days per evaluation, limiting parameter-space exploration. The methods implemented here are general and transferable to other simulation-driven domains.

## Approach

I train a feedforward neural network as a fast surrogate model using simulation data generated for a tokamak plasma system. The surrogate is designed to predict system responses under previously unseen conditions while significantly reducing evaluation time.

Model performance is benchmarked against a linear regression baseline to quantify tradeoffs between predictive accuracy, interpretability, and computational cost. Hyperparameters are optimized using Bayesian optimization (TPE), and robustness is assessed across multiple random seeds.

## Code Structure

src/
- data.py  
  - Data loading, preprocessing, synthetic sampling (Sobol), and persistence utilities.
- model.py  
  - Model definitions, training routines, linear regression baseline, and hyperparameter space.
- train.py  
  - Training orchestration, hyperparameter optimization (TPE), and multi-seed runs.
- eval.py  
  - Evaluation, diagnostics, and visualization of model predictions and optimization results.

## Data

The original simulation data used in this project cannot be shared due to confidentiality constraints. The code is shared to illustrate the workflow. Results reported in the accompanying report are based on real simulation data.

## Results
- The neural network achieves an approximate 15–20% reduction in mean squared error and improves R² from ~0.93 to ~0.94–0.95 on held-out test data, demonstrating improved predictive power while maintaining stability.
- The weakest performance is observed for quantities governed by highly localized and strongly nonlinear effects, suggesting that residual errors are driven by intrinsic system complexity rather than overfitting or numerical issues.
- Overall, the surrogate model delivers orders-of-magnitude speedups over full simulations while preserving predictive accuracy for most targets, making it suitable for rapid parameter exploration and sensitivity analysis.


## How to Run

Run training and optimization:
python run_train.py
Evaluate trained models:
python run_eval.py

## Possible Extensions

- Expand training data coverage by incorporating a larger and more diverse simulation dataset to improve generalization across regimes.
- Enrich the feature set to capture additional dependencies and interactions that are not represented in the current inputs.
- Incorporate domain-informed constraints or structured loss terms to guide the model toward physically consistent solutions.
- Reparameterize inputs and targets using more meaningful aggregate quantities (e.g., fluxes or power-like measures) to simplify the learning task and improve robustness.


## Disclaimer

This project is intended as a methodological demonstration of surrogate modeling and does not represent production software.
