# Different ways to call MultiFactorize. Note the different types of constraints. 

using BlockTensorDecomposition


MultiFactorize = BlockTensorDecomposition.MultiFactorize
generate_tensor = BlockTensorDecomposition.generate_tensor_streams


rank = 10
tensor_size = (20, 10, 200)
Y, matrix, Y_prime = generate_tensor(tensor_size, rank)



decomposition1, stats_data1, kwargs1 = MultiFactorize(Y;
    rank=rank,
    momentum=true,
    tolerence=(0.05),
    converged=(RelativeError),
    constrain_init=true,
    constraints=[simplex_12slices!, simplex_rows!],
    stats=[Iteration, ObjectiveRatio, IterateNormDiff, ObjectiveValue, GradientNNCone, RelativeError, EuclidianLipshitz, EuclidianStepSize]
);


decomposition2, stats_data2, kwargs2 = MultiFactorize(Y;
    rank=rank,
    momentum=true,
    tolerence=(0.05),
    converged=(RelativeError),
    constrain_init=true,
    constraints=[nonnegative!, nonnegative!],
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError, EuclidianLipshitz, EuclidianStepSize]
);


decomposition3, stats_data3, kwargs3 = MultiFactorize(Y;
    rank=rank,
    momentum=true,
    tolerence=(0.05),
    converged=(RelativeError),
    constrain_init=true,
    constraints=[l1scale_12slices! ∘ nonnegative!, l1scale_rows! ∘ nonnegative!],
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError, EuclidianLipshitz, EuclidianStepSize]
);