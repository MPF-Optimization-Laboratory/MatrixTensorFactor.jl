using BlockTensorFactorization

fact = BlockTensorFactorization.factorize

G = abs_randn(2,3,4)
A = abs_randn(10, 2)
B = abs_randn(10, 3)
C = abs_randn(10, 4)
Y = Tucker((G, A, B, C))
Y = array(Y)

decomposition, stats_data, kwargs = fact(Y;
    rank=(2,3,4),
    model=Tucker,
    momentum=true,
    tolerance=(1, 0.045),
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
