# demo of speeding up opnorm using the Gram matrix and Symmetric labeling

using Random: randn
using LinearAlgebra: opnorm, Symmetric
using BenchmarkTools

fopnorm1(X) = opnorm(X)
fopnorm2(X) = sqrt(opnorm(X'X))
fopnorm3(X) = sqrt(opnorm(Symmetric(X'X)))

dimensions = (10000, 10)

b = @benchmark fopnorm1(X) setup=(X=randn(dimensions))
display(b)
b = @benchmark fopnorm2(X) setup=(X=randn(dimensions))
display(b)
b = @benchmark fopnorm3(X) setup=(X=randn(dimensions))
display(b)

dimensions = (100, 100)

b = @benchmark fopnorm1(X) setup=(X=randn(dimensions))
display(b)
b = @benchmark fopnorm2(X) setup=(X=randn(dimensions))
display(b)
b = @benchmark fopnorm3(X) setup=(X=randn(dimensions))
display(b)
