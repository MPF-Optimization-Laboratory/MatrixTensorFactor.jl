using BlockTensorDecomposition

include("../src/SyntheticDataGenerator.jl")

MultiFactorizeSimplex = BlockTensorDecomposition.MultiFactorizeSimplex
fact = BlockTensorDecomposition.factorize 

rank = 3
tensor_size = (5, 3, 16)
Y, matrix, Y_prime = generate_tensor_streams(tensor_size, rank)


@time decomposition1, stats_data1, kwargs1 = MultiFactorizeSimplex(Y;
    rank=rank,
    momentum=false,
    tolerence=(0.05),
    converged=(RelativeError),
    constrain_init=true,
    constraints=[simplex_12slices!, simplex_rows!],#,[l1scale_12slices! ∘ nonnegative!, l1scale_rows! ∘ nonnegative!],
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError, EuclidianLipshitz, EuclidianStepSize]
);


new_core_constraint! = ScaledNormalization(l1norm; whats_normalized=(x -> eachslice(x; dims=1)), scale=size(Y)[2] )
core_constraint_update! = ConstraintUpdate(0, new_core_constraint!; whats_rescaled=(x -> eachcol(matrix_factor(x, 1))))

@time decomposition2, stats_data2, kwargs2 = fact(Y;
    rank=rank,
    momentum=false,
    tolerence=(0.05),
    converged=(RelativeError),
    constrain_init=true,
	constraints = [simplex_12slices!, simplex_rows!],
    #constraints=[core_constraint_update!, ConstraintUpdate(1,  nonnegative!)],
    #constraints=[l1scale_12slices! ∘ nonnegative!, l1scale_rows! ∘ nonnegative!],
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError, EuclidianLipshitz, EuclidianStepSize]
);

