# Context
#This document tests different ways to calculate the operator norm of a matrix in Julia.

#The operator norm is needed to calculate the step-size used in each gradient descent update, so it is important to ensure this operation fast.

# Background

# We define the operator norm of a matrix $A\in\mathbb{R}^{I\times J}$ to be
# $$
# \left\lVert A \right\rVert_{op} = \sup_{\lVert v \rVert_{2}} \left\lVert A v\right\rVert_{2}.
# $$

# It is true that
# $$
# \left\lVert A^\top \right\rVert_{op} = \left\lVert A \right\rVert_{op}
# $$

# and

# $$
# \left\lVert A^\top A \right\rVert_{op} = \left\lVert A A^\top  \right\rVert_{op} = \left\lVert A \right\rVert_{op}^2.
# $$

# # Tests

# Let's look at a few ways we can find the norm of a matrix in Julia. To ensure a fair comparison, we will wrap all of these in a function, and test them out on three types of dense matrices: tall, wide, and square.

## Possible routines
# ```{julia}
# using Pkg; Pkg.activate(".")
# Pkg.add(["LinearAlgebra","Random","Arpack","BenchmarkTools"])
# #Pkg.add(path="C:\\Users\\Nicholas\\OneDrive - UBC\\Research\\general-block-decomposition\\BlockTensorDecomposition.jl")
# ```

using Random: rand, randn
using LinearAlgebra: svdvals, opnorm, Symmetric
using Arpack: svds
using BlockTensorDecomposition: slicewise_dot
using BenchmarkTools


# Now for the test set of functions

fopnorm1(X) = opnorm(X)
fopnorm2(X) = sqrt(opnorm(X'X))
fopnorm3(X) = sqrt(opnorm(X*X'))
fopnorm4(X) = sqrt(opnorm(Symmetric(X'X)))
fopnorm5(X) = sqrt(opnorm(Symmetric(X*X')))

fopnorms = [fopnorm1,fopnorm2,fopnorm3,fopnorm4,fopnorm5]

svdvals_top(X) = svdvals(X)[1]
fsvdvals1(X) = svdvals_top(X)
fsvdvals2(X) = sqrt(svdvals_top(X'X))
fsvdvals3(X) = sqrt(svdvals_top(X*X'))
fsvdvals4(X) = sqrt(svdvals_top(Symmetric(X'X)))
fsvdvals5(X) = sqrt(svdvals_top(Symmetric(X*X')))

fsvdvalss = [fsvdvals1,fsvdvals2,fsvdvals3,fsvdvals4,fsvdvals5]

svds_top(X) = svds(X;nsv=1)[1].S[1]
fsvds1(X) = svds_top(X)
fsvds2(X) = sqrt(svds_top(X'X))
fsvds3(X) = sqrt(svds_top(X*X'))
fsvds4(X) = sqrt(svds_top(Symmetric(X'X)))
fsvds5(X) = sqrt(svds_top(Symmetric(X*X')))

fsvdss = [fsvds1,fsvds2,fsvds3,fsvds4,fsvds5]

#b = @benchmark fopnorm1(A) setup=(A=randn(10,100))

dims = [
    (100, 10),
    (10, 100),
    (100, 100),
    (10, 10)
]

for (N, M) in dims
    A=randn(N, M)
    for f in fopnorms
        @show size(A)
        @show f
        b = @benchmark ($f)($A)
        display(b)
        println()
    end
end

for (N, M) in dims
    A=randn(N, M)
    for f in fsvdvalss
        @show size(A)
        @show f
        b = @benchmark ($f)($A)
        display(b)
        println()
    end
end

for (N, M) in dims
    A=randn(N, M)
    for f in fsvdss
        @show size(A)
        @show f
        b = @benchmark ($f)($A)
        display(b)
        println()
    end
end

# Fastest so far is fopnorm4 or fopnorm5,
# depending on which one gives the smaller sized Gram matrix

# Can we speed it up by reshaping to a tensor, doing a slicewice dot product, and then finding the op norm?

sqrt_int(n) = sqrt(n) |> Int

function fwideslice(X)
    M,N = size(X)
    m = sqrt_int(M)
    Xtensor = reshape(X, m, m, N)
    XX = slicewise_dot(Xtensor,Xtensor)
    opnorm(XX) # has a Symmetric wrapper built in
end

@benchmark fwideslice(B) setup=(B=randn(10,100))

@benchmark fopnorm5(B) setup=(B=randn(10,100))

# The answer is no!
# So we should use fopnorm4 or fopnorm5 always!
