using BlockTensorDecomposition


MultiFactorize = BlockTensorDecomposition.MultiFactorize
fact = BlockTensorDecomposition.factorize 
generate_tensor = BlockTensorDecomposition.generate_tensor_streams


rank = 10
tensor_size = (20, 10, 200)
Y, matrix, Y_prime = generate_tensor(tensor_size, rank)



@time decomposition1, stats_data1, kwargs1 = MultiFactorize(Y;
    rank=rank,
    momentum=true,
    tolerence=(0.05),
    converged=(RelativeError),
    constrain_init=true,
    constraints=[simplex_12slices!, simplex_rows!],
    stats=[Iteration, ObjectiveRatio, IterateNormDiff, ObjectiveValue, GradientNNCone, RelativeError, EuclidianLipshitz, EuclidianStepSize]
);



@time decomposition2, stats_data2, kwargs2 = fact(Y;
    rank=rank,
    momentum=true,
    tolerence=(0.05),
    converged=(RelativeError),
    constrain_init=true,
	constraints = [simplex_12slices!, simplex_rows!],
    maxiter=Inf,
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError, EuclidianLipshitz, EuclidianStepSize]
);

