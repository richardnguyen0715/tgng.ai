# Softmax from Scratch

**Category:** Deep Learning
**Difficulty:** 🟢 Easy

## Problem

Implement the **softmax** function from scratch using only NumPy.

Given a 1-D or 2-D array of real-valued scores (logits), return the softmax probabilities such that:

- Each value is in the range `(0, 1)`.
- Values in each row sum to `1.0`.
- The function is numerically stable (subtract the row-max before exponentiation).

### Function Signature

```python
def softmax(x: np.ndarray) -> np.ndarray:
    ...
```

### Examples

```python
>>> import numpy as np
>>> softmax(np.array([1.0, 2.0, 3.0]))
array([0.09003057, 0.24472847, 0.66524096])

>>> softmax(np.array([[1.0, 2.0], [3.0, 4.0]]))
array([[0.26894142, 0.73105858],
       [0.26894142, 0.73105858]])
```

## Approach

1. For numerical stability, subtract the row maximum from each element before applying `exp`.  
   This avoids overflow for large logits without changing the output (the constants cancel in numerator and denominator).
2. Compute `exp(x - max)` element-wise.
3. Divide by the row sum to normalise.

**Key formula:**

```
softmax(xᵢ) = exp(xᵢ - max(x)) / Σ exp(xⱼ - max(x))
```

## Complexity

| | 1-D input | 2-D input (N × C) |
|---|---|---|
| **Time** | O(C) | O(N · C) |
| **Space** | O(C) | O(N · C) |
