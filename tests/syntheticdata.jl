# generate synthetic data for KDE and tensor factorization

using Random
using KernelDensity
using MatrixTensorFactor
using Plots

Random.seed!(123)

##########
#   1D   #
##########

N = 10 # number of samples
M = 3 # features

list_of_samples = [randn(M) for _ in 1:N]

list_of_measurements = [[n for n in m] for m in zip(list_of_samples...)]

kde(list_of_measurements[1])

kdes = make_densities(list_of_samples)

f = kdes[1] # the first estimated kernel

p = plot(f.x, f.density)
display(p)

##########
#   2D   #
##########

locations = [randn(N, 2) for _ in 1:M]


kdes = make_densities(locations)
f = kdes[1] # the first estimated kernel

p = plot(f.x, f.y, f.density)
display(p)
