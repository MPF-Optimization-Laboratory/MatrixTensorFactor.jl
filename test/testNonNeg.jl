using BlockTensorDecomposition

include("../src/SyntheticDataGenerator.jl")

multifact = BlockTensorDecomposition.MultiFactorize
fact = BlockTensorDecomposition.factorize 


# With these current numbers we can see MultiFact converge but fact doesnt suggesting maybe this boosts convergence rates


rank = 12
size = (15, 12, 512)

Y, matrix, Y_prime = generate_tensor_streams(size, rank)



@time decomposition2, stats_data, kwargs = multifact(Y;
    rank=rank,
    momentum=true,
    tolerence=(0.05),
    converged=(RelativeError),
    constrain_init=true,
    constraints=[nonnegative!, nonnegative!],
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError, EuclidianLipshitz, EuclidianStepSize]
);

@time decomposition2, stats_data, kwargs = fact(Y;
    rank=rank,
    momentum=true,
    tolerence=(0.05),
    converged=(RelativeError),
    constrain_init=true,
    constraints=[nonnegative!, nonnegative!],
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError, EuclidianLipshitz, EuclidianStepSize]
);