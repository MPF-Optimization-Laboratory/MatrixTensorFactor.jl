# This is a test of different approaches to solving the NMF problem
# 1/2 ||AB-Y||_F^2
# where the rows of A, B, & Y all sum to 1

using BlockTensorDecomposition
using Random
using UnicodePlots
using Plots

Random.seed!(31415926535897)

fact = BlockTensorDecomposition.factorize

I, J = 25, 50
R = 5

A = abs_randn(I, R)
l1scale_rows!(A)

B = abs_randn(R, J)
l1scale_rows!(B)

Y = Tucker1((B, A))
Y = array(Y)

@assert check(l1scale_rows!, Y)

A_init = abs_randn(I, R)
l1scale_rows!(A_init)

B_init = abs_randn(R, J)
l1scale_rows!(B_init)

n_iterations = 5000

options = (
    model=Tucker1,
    tolerance=0.0, # Force the algorithm to halt at maxiter
    maxiter=n_iterations,
    converged=RelativeError,
    stats=[
        Iteration, ObjectiveValue, GradientNNCone, RelativeError, #PrintStats
    ],
    momentum=false,
    do_subblock_updates=false,#false#true
)

# Possible updates
# B is the 0th factor, A is the 1st factor

scaleB_rows_rescaleA! = ConstraintUpdate(0, l1scale_rows! ∘ nonnegative!;
    whats_rescaled=(x -> eachcol(factor(x, 1)))
)

scaleB_rows! = ConstraintUpdate(0, l1scale_rows! ∘ nonnegative!;
    whats_rescaled=nothing
)

scaleA_rows! = ConstraintUpdate(1, l1scale_rows! ∘ nonnegative!;
    whats_rescaled=nothing
)

nonnegativeA! = ConstraintUpdate(1, nonnegative!)

nonnegativeB! = ConstraintUpdate(0, nonnegative!)

euclideanProjA! = ConstraintUpdate(1, simplex_rows!)

euclideanProjB! = ConstraintUpdate(0, simplex_rows!)

algorithms = (
    bothSimplex = [euclideanProjB!, euclideanProjA!],
    simplexA = [nonnegativeB!, euclideanProjA!],
    simplexB = [euclideanProjB!, nonnegativeA!],
    bothScale = [scaleB_rows!, scaleA_rows!],
    scaleA = [nonnegativeB!, scaleA_rows!],
    scaleB = [scaleB_rows!, nonnegativeA!], #
    rescaleB_scaleA = [scaleB_rows_rescaleA!, scaleA_rows!],
    rescaleB = [scaleB_rows_rescaleA!, nonnegativeA!],
)

# algorithms = Dict(
#     :bothSimplex => [euclideanProjB!, euclideanProjA!],
#     :simplexA => [nonnegativeB!, euclideanProjA!],
#     :simplexB => [euclideanProjB!, nonnegativeA!],
#     :bothScale => [scaleB_rows!, scaleA_rows!],
#     :scaleA => [nonnegativeB!, scaleA_rows!],
#     :scaleB => [scaleB_rows!, nonnegativeA!], #
#     :rescaleB_scaleA => [scaleB_rows_rescaleA!, scaleA_rows!],
#     :rescaleB => [scaleB_rows_rescaleA!, nonnegativeA!],
# )

# decomposition, stats, kwargs = fact(Y; decomposition=X_init, constraints=algorithms[:bothSimplex], options...)
# lineplot(;xlim=(0,n_iterations), ylim=(1e-1,1e-5))
p = plot(; yscale=:log10)

kwargs = nothing
stats = nothing
for (algorithm, constraints) in pairs(algorithms)
    X_init = Tucker1((copy(B_init), copy(A_init))) # Fresh decomposition since it gets mutated
    # but still the same initialization for A and B

    @time decomposition, stats, kwargs = fact(Y; decomposition=X_init, constraints, options...)

    plot!(p, stats[:,:ObjectiveValue]; linewidth=5, label=String(algorithm))
end

p |> display

ylims!((1e-10,1e-8))
xlims!((4800,5000))

X_init = Tucker1((copy(B_init), copy(A_init))) # Fresh decomposition since it gets mutated
    # but still the same initialization for A and B

@time decomposition, stats, kwargs = fact(Y; decomposition=X_init, constraints=[scaleB_rows_rescaleA!, scaleA_rows!], options..., maxiter=5000)
