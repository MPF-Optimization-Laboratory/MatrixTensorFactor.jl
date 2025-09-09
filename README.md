# BlockTensorDecomposition.jl
BlockTensorDecomposition.jl is a package to factorize tensors. The main feature is its flexibility at decomposing input tensors according to many common tensor models (ex. CP, Tucker) with a number of constraints (ex. nonnegative, simplex), while also supporting user-defined models, constraints, and optimization updates.

# Quick Guide

## Factorizing your data

The main feature of this package is to factorize an array. This is accomplished with the `factorize` function.

Here is an example of an order-3 Tucker-1 factorization.

<img width="500" height="526" alt="tucker_1_example" src="https://github.com/user-attachments/assets/120224d1-f7a9-4566-b416-73917a614ff1" />

Say you've collected your data into an order-3 tensor `Y`. You can use `randn` from `Random.jl` to simulate this.

```julia
using Random

Y = randn(100,100,100)
```

Then you can call `factorize` with a number of keywords. The main keywords you many want to specify are the `model` and `rank`. This lets `factorize` know they type and size of the decomposition. See [Decomposition Models](@ref) for a complete list of avalible models, and how to define your own custom decomposition.

```julia
using BlockTensorDecomposition

X, stats, kwargs = factorize(Y; model=Tucker1, rank=5)
```

## Extracting factors

The main output, `X`, is of type `model` (`Tucker1`). We can call `factors` to see what the decomposed factors are. Or you can call `factor(X,n)` to just extract the `n`th factor. This this case, there are only two factors; the order-3 core (0th factor) `G` and the matrix (1st factor) `A`. The following would all be valid ways to extract these factors.

```julia
typeof(X) == Tucker1{Float64, 3}

# Works on any model
G, A = factors(X)
G = factor(X, 0)
A = factor(X, 1)

# Exclusive to Tucker1, Tucker, and CPDecomposition
G = core(X)
A = matrix_factors(X)[begin]

size(G) == (5, 100, 100)
size(A) == (100, 5)
size(X) == (100, 100, 100)
```

Since the models are subtypes of `AbstractArray`, all the usual array operations can be performed directly on the decomposition.

```julia
# Difference between a Tucker1 type and a regular Array type
X - Y

# Entry X_{123} in the tensor
X[1, 2, 3]
```

If you every want to "flatten" the model into a regular `Array`,  you can call `array`, or multiply the factors yourself.

```julia
Z = array(X)
typeof(Z) == Array{Float64, 3}

G ×₁ A == Z # 1-mode product
nmode_product(G, A, 1) == Z # 1-mode product
mtt(A, G) == Z # matrix times tensor
```

## Iteration Statistics

The output variable `stats` is a `DataFrame` that records the requested stats every iteration. You can pass a list of supported stats, or custom stats. See [`Iteration Stats`](@ref) for more details.

By default, the iteration number, objective value (L2 norm between the input and the model in this case), and the Euclidian norm of the gradient (of the loss function at the current iteration) are recorded. The following would reproduce the default stats in our running example.

```julia
X, stats, kwargs = factorize(Y;
    stats=[Iteration, ObjectiveValue, GradientNorm]
)
```

The full results can be displayed in the REPL, or a vector of the individual stat at some or every iteration can be extracted by using the `Symbol` of that stat (prepend a colon `:` to the name).

```julia
display(stats) # Display the full DataFrame

stats[end, :ObjectiveValue] # Final objective value
stats[:, :ObjectiveValue] # Objective value at every iteration
```

You may also want to see every stat at a particular iteration which can be accessed in the following way. Note that the initilization is stored in the first row, so the nth row stores the stats right *before* the nth iteration, not after.

```julia
stats[begin, :] # Every stat at the initialization
stats[4, :] # Every stat right *before* the 4th iteration
stats[end, :] # Every stat at the final iteration
```

See the `DataFrames.jl` package for more data handeling.

## Output keyword arguments

Since there are many options and a complicated handeling of defaults arguments, the `factorize` function also outputs all the keyword arguments as a `NamedTuple`. This allows you to check what keywords you set, along with the default values that were substituted for the keywords you did not provide.

You can access the values by getting the relevent field, or index (as a `Symbol`). In our running example, this would look like the following.

```julia
kwargs.rank == 5
kwargs[:rank] == 5
getfield(kwargs, :rank) == 5

kwarks.model == Tucker1
kwargs[:model] == Tucker1
getfield(kwargs, :model) == Tucker1
```

# About
(Coming Soon) The package also supports user defined models and constraints provided the operations for combining factor into a tensor, and projecting/applying the constraint are given. It is also a longer term goal to support other optimization objective beyond minimizing the least-squares (Frobenius norm) between the input tensor and model.

The general scheme for computing the decomposition is a generalization of Xu and Yin's Block Coordinate Descent Method (2013) that cyclically updates each factor in a model with a proximal gradient descent step. Note for convex constraints, the proximal operation would be a Euclidean projection onto the constraint set, but we find some improvement with a hybrid approach of a partial Euclidean projection followed by a rescaling step. In the case of a simplex constraint on one factor, this looks like: dividing the constrained factor by the sum of entries, and multiplying another factor by this sum to preserve the product.

# References
Naomi Graham, Nicholas Richardson, Michael P. Friedlander, and Joel Saylor. Tracing Sedimentary Origins in Multivariate Geochronology via Constrained Tensor Factorization. Mathematical Geosciences, Feb. 2025. http://doi.org/10.1007/s11004-024-10175-0

# Related Packages

## For decomposing tensors

- [TensorDecompositions.jl](https://github.com/yunjhongwu/TensorDecompositions.jl): Supports the decompositions; high-order SVD, CP & Tucker (and nonnegative version), symmetric rank-1, and Tensor-CUR. Most models support one or two algorithms (usually alternating methods). No customizability of constraints.
- [NTFk.jl](https://github.com/SmartTensors/NTFk.jl): Only nonnegative Tucker and CP decompositions supported
- [GCPDecompositions.jl](https://github.com/dahong67/GCPDecompositions.jl): Only LBFGSB or ALS algorithms for CPDecompositions
- [NMF.jl](https://github.com/JuliaStats/NMF.jl): Multiple algorithms supported for nonnegative matrix factorizations

## For working with tensors and some basic decompositions

- [TensorOperations.jl](https://github.com/QuantumKitHub/TensorOperations.jl)
- [TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl)
- [TensorToolbox.jl](https://github.com/lanaperisa/TensorToolbox.jl)
- [ITensors.jl](https://github.com/ITensor/ITensors.jl)

## For purely constructing and manipulating tensors

- [Tullio.jl](https://github.com/mcabbott/Tullio.jl): Index notation construction
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl): Index notation construction
- [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl): Fast operations with order 1, 2, and 4 tensors, symmetric tensors supported
- [Tensorial.jl](https://github.com/KeitaNakamura/Tensorial.jl): Statically sized tensors
- [SymmetricTensors.jl](https://github.com/iitis/SymmetricTensors.jl): Working with symmetric tensors
