# generate synthetic data for KDE and tensor factorization

using Random
using KernelDensity
using MatrixTensorFactor

Random.seed!(123)

N = 5 # number of samples
M = 3 # features

list_of_samples = [randn(M) for _ in 1:N]

list_of_measurements = [[n for n in m] for m in zip(list_of_samples...)]

kde(list_of_measurements[1])

kdes = make_densities(list_of_samples)

##########################

locations = [randn(N, 2) for _ in 1:M]


kdes = make_densities(locations)
