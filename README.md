# SVM and Kernel Methods on MNIST (2 vs 6)

Binary digit classification using MNIST (digits 2 and 6). This repo implements SVM variants and kernel methods without using `sklearn.svm`.

## What is included

- Hard-margin SVM (primal): solved as a constrained optimization problem with an explicit bias term.
- Hard-margin SVM (dual): solves for Lagrange multipliers and reconstructs `w` and `b`.
- Gaussian RBF kernel SVM (dual): uses the kernel trick with σ = 1.
- k-nearest neighbors baselines (k = 3 and k = 5).
- Soft-margin SVM (linear and RBF): trained with an SGD style procedure for full training set experiments (C in {1, 3, 5}).
- Kernel k-NN: uses an RBF-kernel induced distance.

Implementation uses general-purpose optimizers (SciPy) and custom code, not a packaged SVM solver.

## Dataset and setup

- Dataset: MNIST.
- Task: binary classification for digits 2 vs 6.
- Labels are mapped to `-1` (digit 2) and `+1` (digit 6).
- Experiments:
  - Hard-margin SVM, kernel SVM, and k-NN: small training subset.
  - Soft-margin SVM and kernel k-NN: full filtered training set.

## Results (0-1 loss)

Small-subset results and full-set baselines are captured directly from the notebook run.

| Model | Setting | Train error | Test error |
|---|---|---:|---:|
| Hard-margin SVM (primal, linear) |  | 0.0000 | 0.019598 |
| Hard-margin SVM (dual, linear) |  | 0.0595 | 0.064824 |
| Hard-margin SVM (dual, RBF kernel) | σ = 1 | 0.0000 | 0.518593 |
| k-NN baseline | k = 3 | 0.0020 | 0.005528 |
| k-NN baseline | k = 5 | 0.0025 | 0.007035 |
| Kernel k-NN | k = 5, σ = 1 | 0.5015 | 0.5186 |

Soft-margin (full train) sweep:

| Model | C | Train error | Test error |
|---|---:|---:|---:|
| Soft SVM (Linear) | 1 | 0.486191 | 0.469347 |
| Soft SVM (RBF, σ=1) | 1 | 0.488296 | 0.481407 |
| Soft SVM (Linear) | 3 | 0.499074 | 0.479397 |
| Soft SVM (RBF, σ=1) | 3 | 0.488296 | 0.481407 |
| Soft SVM (Linear) | 5 | 0.498316 | 0.481407 |
| Soft SVM (RBF, σ=1) | 5 | 0.488296 | 0.481407 |

## How to run

1. Create an environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Open the notebook:
   ```bash
   jupyter lab
   ```
   Then run: `notebooks/svm_kernel_methods_mnist_2v6.ipynb`.

## Repo structure

- `notebooks/`
  - `svm_kernel_methods_mnist_2v6.ipynb`: main implementation and experiments
- `docs/`
  - `problem_statement.md`: short, clean spec for the work

## Notes for readers

- The RBF kernel experiments use σ = 1 to match the experiment setting. In practice, σ and C need tuning for good generalization.
- The soft-margin implementation is an SGD baseline meant for scalability on the full dataset; it is intentionally simple and can be improved with better tuning and feature scaling.
