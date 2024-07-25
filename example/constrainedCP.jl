using BlockTensorDecomposition
using Random
using UnicodePlots

N = 100
R = 5
D = 2

fact = BlockTensorDecomposition.factorize
matricies = [abs_randn(N, R) for _ in 1:D]
l1scaled_cols!.(matricies)
Ydecomp = CPDecomposition(Tuple(matricies))#abs_randn
@assert all(check.(l1scaled_cols!, factors(Ydecomp)))
Y = array(Ydecomp)

X_absrandn = CPDecomposition(ntuple(_ -> N, D), R; init=abs_randn)
X_randexp = CPDecomposition(ntuple(_ -> N, D), R; init=randexp)

options = (
    tolerence=.01,
    maxiter=1000,
    converged=RelativeError,
    constraints=[l1scaled_cols! ∘ nnegative!, simplex_cols!],
    constrain_init=true,
    constrain_output=true,
    momentum=true,
    final_constraints = l1scaled_cols!,
    stats=[
        Iteration, ObjectiveValue, GradientNNCone, RelativeError, FactorNorms, EuclidianLipshitz
    ],
)

decomposition_randn, stats_data_randn, kwargs = fact(Y; decomposition=X_absrandn, options...);

#display(stats_data_randn[:, :RelativeError])
lineplot(log10.(stats_data_randn[:, :RelativeError])) |> display

decomposition_randexp, stats_data_randexp, kwargs = fact(Y; decomposition=X_randexp, options...);
#display(stats_data_randexp[:, :RelativeError])

lineplot(log10.(stats_data_randexp[:, :RelativeError])) |> display

@show stats_data_randn[end, :RelativeError], stats_data_randexp[end, :RelativeError]

display(kwargs[:update])
