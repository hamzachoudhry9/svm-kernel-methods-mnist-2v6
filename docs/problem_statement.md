# Problem statement

Train binary classifiers to separate MNIST digits 2 and 6 (labels mapped to -1 and +1). Implement the following without using `sklearn.svm` or other direct SVM solvers:

1. Hard-margin SVM in the primal form with an explicit bias term.
2. Hard-margin SVM in the dual form, then reconstruct `w` and `b`.
3. Gaussian RBF kernel SVM using the dual formulation (σ = 1).
4. Baseline k-nearest neighbors (k = 3 and k = 5) and compare errors.
5. Soft-margin SVM experiments with C in {1, 3, 5} for both linear and RBF settings (trained with SGD for scalability).
6. Kernel k-NN using an RBF kernel induced distance.

Error is reported using 0-1 loss on both train and test splits.
