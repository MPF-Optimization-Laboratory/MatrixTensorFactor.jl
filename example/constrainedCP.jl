using BlockTensorDecomposition
using Random
using UnicodePlots

N = 10
R = 5
D = 3

fact = BlockTensorDecomposition.factorize
matricies = [abs_randn(N, R) for _ in 1:D]
l1normalize_cols!.(matricies)
Ydecomp = CPDecomposition(Tuple(matricies))#abs_randn
@assert all(check.(l1scaled_cols!, factors(Ydecomp)))
Y = array(Ydecomp)

X_absrandn = CPDecomposition((N,N,N), R; init=abs_randn)
X_randexp = CPDecomposition((N,N,N), R; init=randexp)

decomposition_randn, stats_data_randn, kwargs = fact(Y;
    decomposition=X_absrandn,
    tolerence=.1,
    maxiter=1,
    converged=GradientNNCone,
    constraints=l1scaled_cols! ∘ nnegative!, #[BlockedUpdate([ConstraintUpdate(n, nnegative!), ConstraintUpdate(n, l1scaled_cols!)]) for n in 1:D], #l1scaled_cols!, #[nnegative!, l1scaled_cols!, l1scaled_cols!],
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError, FactorNorms]
);

#display(stats_data_randn[:, :RelativeError])
lineplot(stats_data_randn[:, :RelativeError]) |> display

decomposition_randexp, stats_data_randexp, kwargs = fact(Y;
    decomposition=X_randexp,
    tolerence=.1,
    maxiter=100,
    converged=GradientNNCone,
    constraints=l1scaled_cols!,
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError, FactorNorms]
);
#display(stats_data_randexp[:, :RelativeError])


lineplot(stats_data_randexp[:, :RelativeError]) |> display

display(kwargs[:update])

# using Pkg
# Pkg.add("Plots")
# using Plots
# plot(stats_data[2:end, :EuclidianLipshitz], stats_data[2:end, :EuclidianStepSize])
