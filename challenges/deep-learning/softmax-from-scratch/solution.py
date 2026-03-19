"""
Challenge: Softmax from Scratch
Category:  Deep Learning
Difficulty: Easy

Problem
-------
Implement the softmax function from scratch using only NumPy.

Given a 1-D or 2-D array of real-valued scores (logits), return the softmax
probabilities such that:
  - Each value is in the range (0, 1).
  - Values in each row sum to 1.0.
  - The function is numerically stable.

Approach
--------
Subtract the row maximum before exponentiation to prevent overflow.
The shift cancels out in the softmax formula:

    softmax(xᵢ) = exp(xᵢ - max(x)) / Σ exp(xⱼ - max(x))

Complexity
----------
Time:  O(N · C)  where N = rows, C = columns (or just O(C) for 1-D input)
Space: O(N · C)  for the output array
"""

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    x = np.atleast_2d(x)                       # ensure 2-D for uniform handling
    shifted = x - x.max(axis=1, keepdims=True)  # numerical stability
    exp_x = np.exp(shifted)
    result = exp_x / exp_x.sum(axis=1, keepdims=True)
    return result.squeeze()                     # restore original number of dims


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_1d_output_sums_to_one() -> None:
    x = np.array([1.0, 2.0, 3.0])
    out = softmax(x)
    assert out.shape == (3,), f"Expected shape (3,), got {out.shape}"
    assert np.isclose(out.sum(), 1.0), f"Expected sum 1.0, got {out.sum()}"


def test_1d_known_values() -> None:
    x = np.array([1.0, 2.0, 3.0])
    expected = np.array([0.09003057, 0.24472847, 0.66524096])
    assert np.allclose(softmax(x), expected, atol=1e-6)


def test_2d_row_sums() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = softmax(x)
    assert out.shape == (2, 2), f"Expected shape (2, 2), got {out.shape}"
    assert np.allclose(out.sum(axis=1), [1.0, 1.0])


def test_numerical_stability_large_values() -> None:
    x = np.array([1000.0, 1001.0, 1002.0])
    out = softmax(x)
    # Should not produce NaN or Inf
    assert not np.any(np.isnan(out)), "Output contains NaN"
    assert not np.any(np.isinf(out)), "Output contains Inf"
    assert np.isclose(out.sum(), 1.0)


def test_uniform_input() -> None:
    x = np.array([2.0, 2.0, 2.0])
    expected = np.array([1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(softmax(x), expected)


if __name__ == "__main__":
    test_1d_output_sums_to_one()
    test_1d_known_values()
    test_2d_row_sums()
    test_numerical_stability_large_values()
    test_uniform_input()
    print("All tests passed!")
