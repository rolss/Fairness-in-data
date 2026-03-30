# Penalty-Based Reweighting: Measuring and Mitigating Intersectional Unfairness in Machine Learning

## Main Code
The main experiment pipeline can be found in [Reproducible Pipeline.ipynb](Reproducible%20Pipeline.ipynb).

## Notebook Organization
The notebook is divided into the following sections. You can collapse/expand each heading to improve readability while running experiments.

### 1. Function Declarations
Contains most helper functions used in later cells, including:
- Plotting utilities
- Sample weight computation
- Dataset preparation
- Other experiment utilities

This section usually needs to be run only once per session, especially if you plan to execute multiple tests.

### 2. Experimental Setup
- Parameter Initialization: Automatically initializes experiment parameters. **Add new datasets and models here.**
- Parameter Configuration: The first cell allows manual selection of parameters for the experiments below. Follow the in-cell comments to modify parameters correctly. **Change what subgroup, dataset, model and lambdas are used here.**

### 3. Compute Fairness Metrics and Penalties (Validation and Test)
Computes fairness metrics and penalties for validation and test. These values are used in later sections.

### 4. Distribution Analysis
Plots the distribution of subgroup attribute combinations, including:
- Sample size
- Class imbalance

### 5. Confusion Matrices
Displays confusion matrices for the selected subgroup from both:
- Validation model outputs
- Test model outputs

### 6. Performance Evaluation
Plots model performance before and after reweighting.

### 7. Main Results: Penalty Before vs. After Reweighting
Experimental comparison of penalties before and after reweighting, including:
- Tests across all fairness metrics with a fixed lambda
- Tests across all lambdas with a fixed fairness metric (using both weight formulas)
- 5-fold cross-validation tests across all lambdas with a fixed fairness metric (using both weight formulas)
