# Getting different behavior depending on the assignment being in place

using LinearAlgebra
using Random

A = [1. 2.; 3. 4.]
B = A ./ sum.(eachrow(A))

@show all(sum.(eachrow(B)) .≈ 1) # true

B .= A ./ sum.(eachrow(A))
@show all(sum.(eachrow(B)) .≈ 1) # true

A .= A ./ sum.(eachrow(A))
@show all(sum.(eachrow(A)) .≈ 1) # false
