# generate synthetic data for KDE and tensor factorization

using Random
using KernelDensity
using MatrixTensorFactor
using Plots

Random.seed!(123)

##########
#   2D   #
##########

locations = [randn(N, 2) for _ in 1:M]


kdes = make_densities(locations)
f = kdes[1] # the first estimated kernel

p = plot(f.x, f.y, f.density)
display(p)
