using BlockTensorDecomposition

include("../src/SyntheticDataGenerator.jl")

fact = BlockTensorDecomposition.factorize 
multifact = BlockTensorDecomposition.MultiFactorize


rank = 15
tensor_size = (50, 100, 300)
Y, matrix, Y_prime = generate_tensor_streams(tensor_size, rank)


@time decomposition1, stats_data1, kwargs1 = MultiFactorize(Y;
    rank=rank,
    momentum=true,
    #tolerence=(1.001, 0.06),
    tolerence=(0.06),
    #converged=(ObjectiveRatio, RelativeError),
    converged=(RelativeError),
    constrain_init=true,
    constraints=[simplex_12slices!, simplex_rows!],#,[l1scale_12slices! ∘ nonnegative!, l1scale_rows! ∘ nonnegative!],
    stats=[Iteration, ObjectiveRatio, IterateNormDiff, ObjectiveValue, GradientNNCone, RelativeError, EuclidianLipshitz, EuclidianStepSize]
);


new_core_constraint! = ScaledNormalization(l1norm; whats_normalized=(x -> eachslice(x; dims=1)), scale=size(Y)[2] )
core_constraint_update! = ConstraintUpdate(0, new_core_constraint!; whats_rescaled=(x -> eachcol(matrix_factor(x, 1))))



