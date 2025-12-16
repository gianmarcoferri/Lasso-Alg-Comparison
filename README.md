# Lasso Regression Algorithms Comparison from scratch
This project implements and compares three optimization algorithms for solving Lasso regression on the [California Housing Prices dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices). The algorithms are implemented from scratch in MATLAB and include: ISTA (Iterative Soft-Thresholding Algorithm), ADMM (Alternating Direction Method of Multipliers), and a simulated distributed version of ADMM across multiple agents.

## Algorithms Implemented
1. **ISTA (Iterative Soft-Thresholding Algorithm)**  
   - Solves Lasso regression through iterative gradient descent with soft-thresholding
   - Handles the non-differentiable L1 norm using subdifferential concepts
   - Simple implementation but slower convergence for non-smooth problems

2. **ADMM (Alternating Direction Method of Multipliers)**  
    - Reformulates Lasso with slack variables and solves via variable splitting
   - Alternates between optimizing primal and dual variables
   - Extremely fast convergence with closed-form solutions at each step

3. **Distributed ADMM**  
   - Partitions data across multiple computational agents
   - Each agent optimizes locally while coordinating through a consensus variable
   - Ideal for large-scale or inherently distributed datasets

## Usage
The algorithms are implemented in the `LassoReg` MATLAB class:

```matlab
% Initialize with parameters
lasso = LassoReg(step_size, max_iterations, l1_penalty, tolerance);

% Fit using different algorithms
lasso.fit(X, Y, "ista");      % ISTA
lasso.fit(X, Y, "admm");      % ADMM  
lasso.fit(X, Y, "dist", 8);   % Distributed ADMM with 8 agents
```

## Parameters
- step_size: Learning rate for ISTA, penalty parameter for ADMM (default: 0.01)
- max_iterations: Maximum number of iterations (default: 50000)
- l1_penalty: L1 regularization strength (default: 1)
- tolerance: Convergence threshold (default: 1e-4)
- agents: Number of agents for distributed ADMM (default: 8)

## Convergence Criteria
- ISTA: Stops when ‖w_new - w_old‖ < tolerance
- ADMM: Stops when primal and dual residuals fall below adaptive tolerance thresholds
- Distributed ADMM: Uses consensus-based residuals across all agents

## Result and Performance Comparisons
The algorithms were executed with the following parameters:
- Max iterations = 50000
- Step-size = 0.01
- L1-penalty = 1
- Tolerance = 1e-4
- Agents = 8 (Distributed ADMM)

| Algorithm  | R2     | Time (s) | Iterations |
|------------|--------|----------|------------|
| ISTA       | 0.5339 | 4.3333   | 50000      |
| ADMM       | 0.5793 | 0.0004   | 4          |
| ADMM-Dist  | 0.5794 | 0.0363   | 216        |

## Key Findings
- ADMM demonstrates superior performance, converging in just 4 iterations with the highest R² score
- ADMM-Dist achieves nearly identical accuracy to centralized ADMM but requires more iterations due to coordination overhead
- ISTA shows the slowest convergence, failing to reach the tolerance within the maximum iterations
- The distributed version provides a practical trade-off for scenarios where data is naturally partitioned


