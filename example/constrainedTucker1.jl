using BlockTensorDecomposition

fact = BlockTensorDecomposition.factorize

C = abs_randn(5, 11, 12)
A = abs_randn(10, 5)
Y = Tucker1((C, A))
Y = array(Y)

decomposition, stats_data, kwargs = fact(Y;
    rank=5,
    momentum=true,
    tolerance=(1, 0.03),
    converged=(GradientNNCone, RelativeError),
    constrain_init=true,
    constraints=nonnegative!,
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError, EuclidianLipschitz, EuclidianStepSize]
);

display(stats_data)

display(kwargs[:update])

# using Pkg
# Pkg.add("Plots")
# using Plots
# plot(stats_data[2:end, :EuclidianLipschitz], stats_data[2:end, :EuclidianStepSize])
