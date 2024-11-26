using BlockTensorDecomposition

multifact = BlockTensorDecomposition.MultiFactorize

fact = BlockTensorDecomposition.factorize 

using Distributions
using Random





@time decomposition, stats_data, kwargs = multifact(Y;
    rank=8,
    momentum=true,
    tolerence=(1, 0.03),
    converged=(GradientNNCone, RelativeError),
    constrain_init=true,
    constraints=nonnegative!,
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError, EuclidianLipshitz, EuclidianStepSize]
);

@time decomposition, stats_data, kwargs = fact(Y;
    rank=8,
    momentum=true,
    tolerence=(1, 0.03),
    converged=(GradientNNCone, RelativeError),
    constrain_init=true,
    constraints=nonnegative!,
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError, EuclidianLipshitz, EuclidianStepSize]
);