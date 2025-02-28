using BlockTensorDecomposition
using Random

N = 100
R = 5
D = 2

fact = BlockTensorDecomposition.factorize
matrices = [abs_randn(N, R) for _ in 1:D]
l1scale_cols!.(matrices)
Ydecomp = CPDecomposition(Tuple(matrices))#abs_randn
@assert all(check.(l1scale_cols!, factors(Ydecomp)))
Y = array(Ydecomp)

X = CPDecomposition(ntuple(_ -> N, D), R)

options = (
    tolerance=.01,
    decomposition=X,
    maxiter=1000,
    converged=RelativeError,
    constraints=[l1scale_cols! ∘ nonnegative!, simplex_cols!],
    constrain_init=true,
    constrain_output=true,
    final_constraints = l1scale_cols!,
    stats=[
        Iteration, ObjectiveValue, GradientNNCone, RelativeError, FactorNorms, EuclidianLipschitz
    ],
)

# Test initialization functions of factorize
kwargs = BlockTensorDecomposition.default_kwargs(Y; options...)

decomposition, kwargs = BlockTensorDecomposition.initialize_decomposition(Y; kwargs...)
previous, updateprevious! = BlockTensorDecomposition.initialize_previous(decomposition, Y; kwargs...)
parameters, updateparameters! = BlockTensorDecomposition.initialize_parameters(decomposition, Y, previous; kwargs...)
update!, kwargs = BlockTensorDecomposition.make_update!(decomposition, Y; kwargs...)
