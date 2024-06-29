# Quick Guide

## Factorizing your data

The main feature of this package is to factorize an array. This is accomplished with the `factorize` function.

Say you've collected your data into an order-$3$ tensor `Y`. You can use `randn` from `Random.jl` to simulate this.

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
G, A = factors(X)
G = factor(X, 0)
A = factor(X, 1)
G = core(X)
A = matrix_factors(X)[begin]

size(G) == (5, 100, 100)
size(A) == (100, 5)
size(X) == (100, 100, 100)
```

You can flatten the model into a regular `Array` type by calling `array`, or multiplying the factors yourself.

```julia
Z = array(X)
typeof() == Array{Float64, 3}

G ×₁ A == Z # 1-mode product
nmode_product(G, A, 1) == Z # 1-mode product
mtt(A, G) == Z # matrix times tensor
```

## Looking at iterations

The output variable `stats` is a `DataFrame` that details a number of stats every iteration. My default, these are
