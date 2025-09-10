using BlockTensorFactorization
using Random

fact = BlockTensorFactorization.factorize

# Generate two matrices with columns constrained to the simplex
I, J = 10, 20
R = 3

A = abs.(randn(I, R))
B = abs.(randn(R, J))

l1scale_cols!(A) # ensures columns sum to 1 since A is already nonnegative
l1scale_cols!(B) # simplex_cols! can make the factors too sparse

Ydecomp = Tucker1((B, A)) # This means Y = A*B

@assert Ydecomp ≈ A*B
@assert all(check.(simplex_cols!, factors(Ydecomp)))

Y = array(Ydecomp) # Convert to a plain matrix type

# Set options for factorization. Enter the following in the REPL for full list of options
# julia>?BlockTensorFactorization.Core.default_kwargs

options = (
    rank=R,
    model=Tucker1,
    momentum=false, # momentum causes slower iterations, but often fewer total iterations
    do_subblock_updates=true, # uses independent stepsizes on each row, col, or fibre. Can speed up convergence but possibly unstable if an iterate is near the boundary of the constraints
    tolerance=.01, # want less than 1% error
    converged=RelativeError,
    maxiter=2000,
    stats=[
        Iteration, ObjectiveValue, GradientNNCone, RelativeError,
    ],

    # Constrain columns of A and B to the simplex.
    # If A has simplex constrained columns and B is only nonnegative,
    # use [simplex_cols!, nonnegative!] or [l1scale_cols! ∘ nonnegative!, nonnegative!]
    # Can also try [simplex_cols!, simplex_cols!] to constrain A via
    # (Euclidean) simplex projections rather than nonnegative followed by a rescaling,
    # but is usually slower than l1scale_cols! ∘ nonnegative!.

    constraints=[l1scale_cols! ∘ nonnegative!, l1scale_cols! ∘ nonnegative!],
    #constraints=[simplex_cols!, simplex_cols!],

    # Ensure the initialization satisfies the constraints
    constrain_init=true,

    # Ensure the output satisfies the constraints given by `final_constraints`.
    # If `final_constraints` is not given, fallback to using `constraints`.
    # This will add one extra iteration in the stats
    constrain_output=true,
    final_constraints = [l1scale_cols! ∘ nonnegative!, l1scale_cols! ∘ nonnegative!],
)

# Factorize Y

decomposition, stats_data, kwargs = fact(Y; options...);

display(stats_data)

B_learned, A_learned = factors(decomposition)

# Check the two factors multiply to Y and satisfy the constraints

@assert isapprox(Y, A_learned*B_learned; rtol=0.02) # might be slightly more than 1% after `constrain_output`
@assert all(check.(simplex_cols!, (B_learned, A_learned)))
