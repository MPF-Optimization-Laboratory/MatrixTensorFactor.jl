# Block Tensor Decomposition

```@contents
Depth = 3
```

## About this package
`BlockTensorFactorization.jl` is a package to factorize tensors. The main feature is its flexibility at decomposing input tensors according to many common tensor models (ex. CP, Tucker) with a number of constraints (ex. nonnegative, simplex).

(Coming Soon) The package also supports user defined models and constraints provided the operations for combining factor into a tensor, and projecting/applying the constraint are given. It is also a longer term goal to support other optimization objective beyond minimizing the least-squares (Frobenius norm) between the input tensor and model.

The general scheme for computing the decomposition is a generalization of Xu and Yin's Block Coordinate Descent Method (2013) that cyclicaly updates each factor in a model with a proximal gradient descent step. Note for convex constraints, the proximal operation would be a Euclidian projection onto the constraint set, but we find some improvment with a hybrid approach of a partial Euclidian projection followed by a rescaling step. In the case of a simplex constraint on one factor, this looks like: dividing the constrained factor by the sum of entries, and multiplying another factor by this sum to preserve the product.
