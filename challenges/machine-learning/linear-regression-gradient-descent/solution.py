"""
Challenge: Linear Regression with Gradient Descent
Category:  Machine Learning
Difficulty: Easy

Problem
-------
Implement Linear Regression using batch gradient descent from scratch.

Given N training samples (X, y), minimise the Mean Squared Error loss:

    L(w, b) = (1/N) · Σ (ŷᵢ - yᵢ)²   where ŷᵢ = w · xᵢ + b

Return the learned weight vector w and bias b.

Approach
--------
1. Initialise w = zeros(D), b = 0.
2. Forward pass: ŷ = X @ w + b
3. Gradients:
       dw = (2/N) · Xᵀ · (ŷ - y)
       db = (2/N) · sum(ŷ - y)
4. Update: w -= lr * dw,  b -= lr * db
5. Repeat for `epochs` iterations.

Complexity
----------
Time:  O(epochs · N · D)
Space: O(N · D)
"""

import numpy as np


def linear_regression_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    epochs: int = 1000,
) -> tuple[np.ndarray, float]:
    """Fit linear regression via batch gradient descent.

    Parameters
    ----------
    X      : shape (N, D) — feature matrix
    y      : shape (N,)   — target values
    lr     : learning rate
    epochs : number of gradient descent iterations

    Returns
    -------
    w : shape (D,) — learned weight vector
    b : float      — learned bias
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        y_pred = X @ w + b          # (N,)
        error = y_pred - y          # (N,)

        dw = (2 / n) * (X.T @ error)   # (D,)
        db = (2 / n) * error.sum()      # scalar

        w -= lr * dw
        b -= lr * db

    return w, float(b)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_perfect_fit_no_noise() -> None:
    """Should recover exact coefficients when there is no noise."""
    np.random.seed(0)
    X = np.random.randn(200, 1)
    y = 2.0 * X.squeeze() + 7.0
    w, b = linear_regression_gd(X, y, lr=0.1, epochs=2000)
    assert np.isclose(w[0], 2.0, atol=1e-3), f"w={w[0]}"
    assert np.isclose(b, 7.0, atol=1e-3), f"b={b}"


def test_noisy_data() -> None:
    """Should be close to true coefficients with small noise."""
    np.random.seed(42)
    X = np.random.randn(500, 1)
    y = 3.0 * X.squeeze() + 5.0 + np.random.randn(500) * 0.1
    w, b = linear_regression_gd(X, y, lr=0.1, epochs=2000)
    assert np.isclose(w[0], 3.0, atol=0.05), f"w={w[0]}"
    assert np.isclose(b, 5.0, atol=0.05), f"b={b}"


def test_multi_feature() -> None:
    """Should handle D > 1."""
    np.random.seed(7)
    true_w = np.array([1.5, -2.0, 0.5])
    X = np.random.randn(300, 3)
    y = X @ true_w + 1.0
    w, b = linear_regression_gd(X, y, lr=0.05, epochs=3000)
    assert np.allclose(w, true_w, atol=0.01), f"w={w}"
    assert np.isclose(b, 1.0, atol=0.01), f"b={b}"


def test_output_shapes() -> None:
    X = np.random.randn(50, 4)
    y = np.random.randn(50)
    w, b = linear_regression_gd(X, y)
    assert w.shape == (4,)
    assert isinstance(b, float)


if __name__ == "__main__":
    test_perfect_fit_no_noise()
    test_noisy_data()
    test_multi_feature()
    test_output_shapes()
    print("All tests passed!")
