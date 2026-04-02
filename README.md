# tgng.ai 🤖

A personal repository for daily AI/ML coding challenge solutions — like LeetCode, but for Artificial Intelligence and Machine Learning.

## 📖 About

This repo tracks my daily practice solving AI/ML challenges covering topics such as:

- **Machine Learning** — linear/logistic regression, SVMs, clustering, feature engineering
- **Deep Learning** — neural networks, backpropagation, CNNs, RNNs, transformers
- **Natural Language Processing** — tokenization, embeddings, text classification, seq2seq
- **Computer Vision** — image processing, object detection, segmentation
- **Reinforcement Learning** — Q-learning, policy gradients, environments
- **Data Structures & Algorithms** — matrix ops, graph algorithms, dynamic programming for AI

Each solution includes the problem description, approach/intuition, and well-commented code.

## 📁 Folder Structure

```
tgng.ai/
├── challenges/
│   ├── machine-learning/
│   ├── deep-learning/
│   ├── nlp/
│   ├── computer-vision/
│   ├── reinforcement-learning/
│   └── data-structures-algorithms/
├── templates/
│   └── solution_template.py
└── README.md
```

Each challenge lives in its own folder under the relevant category:

```
challenges/<category>/<challenge-name>/
├── README.md        # Problem statement & approach
└── solution.py      # Solution code
```

## 🚀 Getting Started

```bash
# Clone the repo
git clone <repo-url>
cd tgng.ai

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the common dependency used by most solutions
pip install numpy
```

## 📝 Solution Template

Use the template in `templates/solution_template.py` when adding a new challenge solution.

## 📊 Progress Tracker

| # | Challenge | Category | Difficulty | Status |
|---|-----------|----------|------------|--------|
| 1 | [Softmax from Scratch](challenges/deep-learning/softmax-from-scratch) | Deep Learning | Easy | ✅ |
| 2 | [Linear Regression with Gradient Descent](challenges/machine-learning/linear-regression-gradient-descent) | Machine Learning | Easy | ✅ |

## 🏷️ Difficulty Legend

| Label | Meaning |
|-------|---------|
| 🟢 Easy | Foundational concepts, straightforward implementation |
| 🟡 Medium | Requires solid understanding, some tricky parts |
| 🔴 Hard | Complex architectures, advanced math, or optimization required |

---

*Inspired by LeetCode — but for the AI/ML world.*
