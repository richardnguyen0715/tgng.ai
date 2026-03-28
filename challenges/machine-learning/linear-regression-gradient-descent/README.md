# Linear Regression with Gradient Descent

**Category:** Machine Learning
**Difficulty:** 🟢 Easy

## Problem

Implement **Linear Regression** using **batch gradient descent** from scratch with NumPy.

Given a dataset of `N` training samples with `D` features each, fit the model parameters `w` (weights) and `b` (bias) by minimising the **Mean Squared Error (MSE)** loss:

```
L(w, b) = (1 / N) · Σ (ŷᵢ - yᵢ)²
```

where `ŷᵢ = w · xᵢ + b`.

### Function Signature

```python
def linear_regression_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    epochs: int = 1000,
) -> tuple[np.ndarray, float]:
    ...
```

Returns `(w, b)` after training.

### Example

```python
>>> import numpy as np
>>> np.random.seed(42)
>>> X = np.random.randn(100, 1)
>>> y = 3 * X.squeeze() + 5 + np.random.randn(100) * 0.1
>>> w, b = linear_regression_gd(X, y, lr=0.1, epochs=1000)
>>> round(float(w[0]), 1), round(b, 1)
(3.0, 5.0)
```

## Approach

1. **Initialise** weights `w = zeros(D)` and bias `b = 0`.
2. **Forward pass** — compute predictions: `ŷ = X @ w + b`.
3. **Compute gradients** (partial derivatives of MSE):
   - `∂L/∂w = (2/N) · Xᵀ · (ŷ - y)`
   - `∂L/∂b = (2/N) · Σ(ŷ - y)`
4. **Update** parameters: `w -= lr · ∂L/∂w`, `b -= lr · ∂L/∂b`.
5. Repeat for `epochs` iterations.

## Complexity

| | Value |
|---|---|
| **Time per epoch** | O(N · D) |
| **Space** | O(N · D) |
