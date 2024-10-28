# Code for using the the new factorize internals with the old interface
#
# Wraps the new `factorize` function with nnmtf so that old code doesn't break,
# but can still benifit from the newer and more optimized internals.

# nnmtf can still call _nnmtf_proxgrad as before
# We just need to change the internals

#---------------------------- data ------------------------------------#
struct UnimplimentedError <: Exception end

"""
    IMPLIMENTED_NORMALIZATIONS::Set{Symbol}

- `:fibres`: set ``\\sum_{k=1}^K B[r,j,k] = 1`` for all ``r, j``, or when `projection==:nnscale`,
    set ``\\sum_{j=1}^J\\sum_{k=1}^K B[r,j,k] = J`` for all ``r``
- `:slices`: set ``\\sum_{j=1}^J\\sum_{k=1}^K B[r,j,k] = 1`` for all ``r``
- `:nothing`: does not enforce any normalization of `B`
- `:rows`: set a metric on eachslice(X; dims=1) to 1, which is equivelent to eachrow(X) when X is a matrix
- `:cols`: set a metric on eachslice(X; dims=2) to 1, which is equivelent to eachcol(X) when X is a matrix
"""
const IMPLIMENTED_NORMALIZATIONS = Set{Symbol}((:fibres, :slices, :rows, :cols, :nothing))

"""
    IMPLIMENTED_METRICS::Set{Symbol}

- `:L1`: the default; ensure sums of entries in each fibre or slice (according to `normalize`)
    are equal to 1
- TODO `:L2`: ensure the sum of squares of entries are 1
- `:Linfty`: ensures the maximum entry is 1
- `:nothing`: do not enforce a metric to equal 1
"""
const IMPLIMENTED_METRICS = Set{Symbol}((:L1, :Linfinity, :nothing)) # TODO :L2

"""
    IMPLIMENTED_PROJECTIONS::Set{Symbol}

- `:nnscale`: Two stage block coordinate decent; 1) projected gradient decent onto nonnegative
    orthant, 2) shift any weight from `B` to `A` according to normalization. Equivilent to
    :nonnegative when `normalization==:nothing`.
- `:simplex`: Euclidian projection onto the simplex blocks accoring to `normalization`
- `:nonnegative`: zero out negative entries
"""
const IMPLIMENTED_PROJECTIONS = Set{Symbol}((:nnscale, :simplex, :nonnegative)) # nn is nonnegative

"""
    IMPLIMENTED_CRITERIA::Set{Symbol}

- `:ncone`: vector-set distance between the -gradient of the objective and the normal cone
- `:iterates`: A,B before and after one iteration are close in L2 norm
- `:objective`: objective is small
- `:relativeerror`: relative error is small (when `normalize=:nothing`) or
    mean relative error averaging fibres or slices when the normalization is `:fibres` or
    `:slices` respectfuly.
"""
const IMPLIMENTED_CRITERIA = Set{Symbol}((:ncone, :iterates, :objective, :relativeerror))

"""
    IMPLIMENTED_STEPSIZES::Set{Symbol}

- `:lipshitz`: gradient step 1/L for lipshitz constant L
- `:spg`: spectral projected gradient stepsize
"""
const IMPLIMENTED_STEPSIZES = Set{Symbol}((:lipshitz, :spg))

"""Lists all implimented options"""
const IMPLIMENTED_OPTIONS = Dict(
    "normalizations" => IMPLIMENTED_NORMALIZATIONS,
    "projections" => IMPLIMENTED_PROJECTIONS,
    "criteria" => IMPLIMENTED_CRITERIA,
    "stepsizes" => IMPLIMENTED_STEPSIZES,
)

const parse_criterion = (
    ncone = GradientNNCone,
    iterates = Iteration,
    objective = ObjectiveValue,
    relativeerror = RelativeError,
)

const normalize_to_simplex_constraint = (
    fibres = simplex_12slices!,
    slices = simplex_1slices!,
    rows = simplex_rows!,
    cols = simplex_cols!,
)

const normalize_to_scaled_l1_constraint = (
    fibres = l1scale_average12slices! ∘ nonnegative!,
    slices = l1scale_1slices! ∘ nonnegative!,
    rows = l1scale_rows! ∘ nonnegative!,
    cols = l1scale_cols! ∘ nonnegative!,
)

const normalize_to_scaled_linfty_constraint = (
    fibres = linftyscale_average12slices! ∘ nonnegative!,
    slices = linftyscale_1slices! ∘ nonnegative!,
    rows = linftyscale_rows! ∘ nonnegative!,
    cols = linftyscale_cols! ∘ nonnegative!,
)

function parse_normalization_projection(normalize, projection, metric)
    # Possible projections
    #(:nnscale, :simplex, :nonnegative)

    # Possible metrics
    #(:L1, :Linfinity, :nothing)

    # Possible normalizations
    #(:fibres, :slices, :rows, :cols, :nothing)

    if projection == :nonnegative || normalize == :nothing
        return nonnegative!

    elseif projection == :simplex
        if metric == :L1
            return normalize_to_simplex_constraint[normalize]
        else
            return throw(UnimplimentedError("The combination of projection $projection and metric $metric is not implemented (YET!)"))
        end

    elseif projection == :nnscale
        if metric == :L1
            return normalize_to_scaled_l1_constraint[normalize]
        elseif metric == :Linfinity
            return normalize_to_scaled_linfty_constraint[normalize]
        else
            return throw(UnimplimentedError("The combination of projection $projection and metric $metric is not implemented (YET!)"))
        end

    else
        return throw(UnimplimentedError("The combination of projection $projection and metric $metric is not implemented (YET!)"))
    end
end


"""
Default initialization
"""
_init(x...) = abs.(randn(x...))

#---------------------------- nnmtf ------------------------------------#

@doc raw"""
    nnmtf(Y::AbstractArray, R::Integer; kwargs...)

Non-negatively matrix-tensor factorizes an order N tensor Y with a given "rank" R.

For an order ``N=3`` tensor, this factorizes ``Y \approx A B`` where
``\displaystyle Y[i,j,k] \approx \sum_{r=1}^R A[i,r]*B[r,j,k]``
and the factors ``A, B \geq 0`` are nonnegative.

For higher orders, this becomes
``\displaystyle Y[i1,i2,...,iN] \approx \sum_{r=1}^R A[i1,r]*B[r,i2,...,iN].``

Note there may NOT be a unique optimal solution

# Arguments
- `Y::AbstractArray{T,N}`: tensor to factorize
- `R::Integer`: rank to factorize Y (size(A)[2] and size(B)[1])

# Keywords
- `maxiter::Integer=100`: maxmimum number of iterations
- `tol::Real=1e-3`: desiered tolerance for the convergence criterion
- `rescale_AB::Bool=true`: scale B at each iteration so that the factors (horizontal slices) have similar 3-fiber sums.
- `rescale_Y::Bool=true`: Preprocesses the input `Y` to have normalized 3-fiber sums (on average), and rescales the final `B` so `Y=A*B`.
- `normalize::Symbol=:fibres`: part of B that should be normalized (must be in IMPLIMENTED_NORMALIZATIONS)
- `normalizeA::Symbol=:rows`: part of A that should be normalized (must be in IMPLIMENTED_NORMALIZATIONS)
- `projection::Symbol=:nnscale`: constraint to use and method for enforcing it (must be in IMPLIMENTED_PROJECTIONS)
- `metric::Symbol=:L1`: under what metric the fibres/slices are normalized (must be in IMPLIMENTED_METRICS)
- `criterion::Symbol=:ncone`: how to determine if the algorithm has converged (must be in IMPLIMENTED_CRITERIA)
- `stepsize::Symbol=:lipshitz`: used for the gradient decent step (must be in IMPLIMENTED_STEPSIZES)
- `momentum::Bool=false`: use momentum updates
- `delta::Real=0.9999`: safeguard for maximum amount of momentum (see eq (3.5) Xu & Yin 2013)
- `R_max::Integer=size(Y)[1]`: maximum rank to try if R is not given
- `projectionA::Symbol=projection`: projection to use on factor A (must be in IMPLIMENTED_PROJECTIONS)
- `projectionB::Symbol=projection`: projection to use on factor B (must be in IMPLIMENTED_PROJECTIONS)
- `metricA::Symbol=metric`: the metric to use for factor A (must be in IMPLIMENTED_METRICS)
- `metricB::Symbol=metric`: the metric to use for factor B (must be in IMPLIMENTED_METRICS)
- `scaleBtoA::Bool=true`: when using `projection=:nnscale`, if the weights should be moved from B to A, or A to B
- `A_init::AbstractMatrix=nothing`: initial A for the iterative algorithm. Should be kept as nothing if `R` is not given.
- `B_init::AbstractArray=nothing`: initial B for the iterative algorithm. Should be kept as nothing if `R` is not given.

# Returns
- `A::Matrix{Float64}`: the matrix A in the factorization Y ≈ A * B
- `B::Array{Float64, N}`: the tensor B in the factorization Y ≈ A * B
- `rel_errors::Vector{Float64}`: relative errors at each iteration
- `norm_grad::Vector{Float64}`: norm of the full gradient at each iteration
- `dist_Ncone::Vector{Float64}`: distance of the -gradient to the normal cone at each iteration
- If R was estimated, also returns the optimal `R::Integer`

# Implimentation of block coordinate decent updates
We calculate the partial gradients and corresponding Lipshitz constants like so:

```math
\begin{align}
  \boldsymbol{P}^{t}[q,r] &=\textstyle{\sum}_{jk} \boldsymbol{\mathscr{B}}^n[q,j,k] \boldsymbol{\mathscr{B}}^n[r,j,k]\\
  \boldsymbol{Q}^{t}[i,r] &=\textstyle{\sum}_{jk}\boldsymbol{\mathscr{Y}}[i,j,k] \boldsymbol{\mathscr{B}}^n[r,j,k] \\
  \nabla_{A} f(\boldsymbol{A}^{t},\boldsymbol{\mathscr{B}}^{t}) &= \boldsymbol{A}^{t} \boldsymbol{P}^{t} - \boldsymbol{Q}^{t} \\
  L_{A} &= \left\lVert \boldsymbol{P}^{t} \right\rVert_{2}.
\end{align}
```

Similarly for `` \boldsymbol{\mathscr{B}}``:

```math
\begin{align}
  \boldsymbol{T}^{t+1}&=(\boldsymbol{A}^{t+\frac12})^\top \boldsymbol{A}^{t+\frac12}\\
  \boldsymbol{\mathscr{U}}^{t+1}&=(\boldsymbol{A}^{t+\frac12})^\top \boldsymbol{\mathscr{Y}} \\
  \nabla_\boldsymbol{\mathscr{B}} f(\boldsymbol{A}^{t+\frac12},\boldsymbol{\mathscr{B}}^{t}) &=  \boldsymbol{T}^{t+1} \boldsymbol{\mathscr{B}}^{t} - \boldsymbol{\mathscr{U}}^{t+1} \\
  L_B &= \left\lVert \boldsymbol{T}^{t+1} \right\rVert_{2}.
\end{align}
```

To ensure the iterates stay "close" to normalized, we introduce a renormalization step after
the projected gradient updates:

```math
\begin{align}
    \boldsymbol{C} [r,r]&=\frac{1}{J}\textstyle{\sum}_{jk} \boldsymbol{\mathscr{B}}^{t+\frac12}[r,j,k]\\
    \boldsymbol{A}^{t+1}&= \boldsymbol{A}^{t+\frac12} \boldsymbol{C}\\
    \boldsymbol{\mathscr{B}}^{t+1}&= (\boldsymbol{C}^{t+1})^{-1}\boldsymbol{\mathscr{B}}^{t+\frac12}.
\end{align}
```

We typicaly use the following convergence criterion:
```math
d(-\nabla \ell(\boldsymbol{A}^{t},\boldsymbol{\mathscr{B}}^{t}), N_{\mathcal{C}}(\boldsymbol{A}^{t},\boldsymbol{\mathscr{B}}^{t}))^2\leq\delta^2 R(I+JK).
```
"""
function nnmtf(Y::AbstractArray, R::Union{Nothing, Integer}=nothing;
    normalize::Symbol=:fibres,
    projection::Symbol=:nnscale,
    criterion::Symbol=:ncone,
    stepsize::Symbol=:lipshitz,
    momentum::Bool=false,
    R_max::Integer=size(Y)[1], # Number of observed mixtures
    online_rank_estimation::Bool=false,
    kwargs...
)

    # Iteration option checking
    if !(projection in IMPLIMENTED_PROJECTIONS)
        throw(ArgumentError("projection is not an implimented projection"))
    elseif !(normalize in IMPLIMENTED_NORMALIZATIONS)
        throw(ArgumentError("normalize is not an implimented normalization"))
    elseif !(criterion in IMPLIMENTED_CRITERIA)
        throw(ArgumentError("criterion is not an implimented criterion"))
    elseif !(stepsize in IMPLIMENTED_STEPSIZES)
        throw(ArgumentError("stepsize is not an implimented stepsize"))
    end

    if momentum && stepsize != :lipshitz
        throw(ArgumentError("Momentum is only compatible with lipshitz stepsize"))
    end

    if isnothing(R) #&& (online_rank_estimation == true)
        # Run nnmtf with R from 1 to size(Y)[1]
        # Compare fit ||Y - AB||_F^2 across all R
        # Return the output at the maximum positive curavature of ||Y - AB||_F^2
        all_outputs = []
        final_rel_errors = Real[]
        @info "Estimating Rank"
        for r in 1:R_max
            @info "Trying rank=$r..."
            output = _nnmtf_proxgrad(Y, r; normalize, projection, criterion, stepsize, momentum, kwargs...)
            push!(all_outputs, output)
            final_rel_error = output[3][end]
            push!(final_rel_errors, final_rel_error)
            @info "Final relative error = $final_rel_error"

            if (online_rank_estimation == true) && length(final_rel_errors) >= 3 # Need at least 3 points to evaluate curvature
                curvatures = standard_curvature(final_rel_errors)
                if curvatures[end] ≈ maximum(curvatures) # want the last curvature to be significantly smaller than the max
                    continue
                else
                    R = argmax(curvatures)
                    @info "Optimal rank found: $R"
                    return ((all_outputs[R])..., R)
                end
            end
        end
        R = argmax(standard_curvature(final_rel_errors))
        @info "Optimal rank found: $R"
        return ((all_outputs[R])..., R)
    end

    return _nnmtf_proxgrad(Y, R; normalize, projection, criterion, stepsize, momentum, kwargs...)
end

function _nnmtf_proxgrad(
    Y::AbstractArray,
    R::Integer;
    maxiter::Integer=1000,
    tol::Real=1e-4,
    normalize::Symbol=:fibres,
    normalizeA::Symbol=:rows,
    projection::Symbol=:nonnegative,
    metric::Symbol=:L1,
    stepsize::Symbol=:lipshitz,
    criterion::Symbol=:ncone,
    momentum::Bool=false,
    delta::Real=0.9999,
    rescale_AB::Bool = (projection == :nnscale ? true : false),
    rescale_Y::Bool = (projection == :nnscale ? true : false),
    projectionA::Symbol = projection,
    projectionB::Symbol = projection,
    metricA::Symbol = metric,
    metricB::Symbol = metric,
    scaleBtoA::Bool = true,
    A_init::Union{Nothing, AbstractMatrix}=nothing,
    B_init::Union{Nothing, AbstractArray}=nothing,
)
    #--- Legacy argument parsing ---#
    # Override scaling if no normalization is requested
    normalize == :nothing ? (rescale_AB = rescale_Y = false) : nothing

    # Extract Dimentions
    M, Ns... = size(Y)

    # Initialize A, B
    if A_init === nothing
        A = _init(M, R)
    else
        size(A_init) == (M, R) || throw(ArgumentError("A_init should have size $((M, R)), got $(size(A_init))"))
        A = A_init
    end

    if B_init === nothing
        B = _init(R, Ns...)
    else
        size(B_init) == (R, Ns...) || throw(ArgumentError("A_init should have size $((R, Ns...)), got $(size(B_init))"))
        B = B_init
    end

    # Only want to rescale the initialization if both A and B were not given
    # Otherwise, we should use the provided initialization
    if rescale_AB && A_init === nothing && B_init === nothing
        rescaleAB!(A, B; normalize, metricA, metricB, scaleBtoA)
    end

    problem_size = R*(M + prod(Ns))

    # # Scale Y if desired
    if rescale_Y
        # Y_input = copy(Y)
        Y, factor_sums = rescaleY(Y; normalize, metric)
    end

    #--- Additional Parse of arguments ---#
    normalizeB = normalize
    if parse_criterion == :ncone
        tol /= sqrt(problem_size) # match old tolerance
    end
    (stepsize == :lipshitz) || throw(ArgumentError("stepsize $stepsize is not implimented"))
    (scaleBtoA == true) || @warn "scaleBtoA==false was considered in initialization, but not during iteration"

    #--- Transform them into something factorize can take ---#
    constraintA = parse_normalization_projection(normalizeA, projectionA, metricA)
    constraintB = parse_normalization_projection(normalizeB, projectionB, metricB)

    if projectionB == :nnscale # move the weights from the tensor B to the matrix A
        constraintB = ConstraintUpdate(0, constraintB; whats_rescaled=(x -> eachcol(matrix_factor(x, 1))))
        constraintA = ConstraintUpdate(1, constraintA) # need to wrap constraintA to match type of constraintB
    end

    factorize_criterion = parse_criterion[criterion]
    constrain_output = false
    if allequal((metric,metricA,metricB))
        constrain_output = (metric == :L1)
    else
        @warn "(metric, metricA, metricB) = $((metric,metricA,metricB)) are not all the same, setting constrain_output=false"
    end
    decomposition = Tucker1((B, A))

    #--- output = factorize(input) ---#
    X, stats, kwargs = factorize(Y;
        model=Tucker1,
        decomposition,
        rank=R,
        stats=[Iteration, RelativeError, GradientNorm, GradientNNCone, ObjectiveValue],#
        constraints=[constraintB, constraintA],
        converged=factorize_criterion,
        maxiter,
        tolerence=tol,
        momentum,
        δ=delta,
        constrain_output,
        constrain_init=true, # new to this version
    )

    #--- Process output to return the same types as the old nntf ---#
    # Use collect to ensure they are all plain array types
    A = matrix_factor(X, 1) |> collect
    B = core(X) |> collect
    rel_errors = stats[:, Symbol(RelativeError)] |> collect
    norm_grad = stats[:, Symbol(GradientNorm)] |> collect
    dist_Ncone = stats[:, Symbol(GradientNNCone)] |> collect

    #--- Old post processing ---#
    # Rescale B back if Y was initialy scaled
    # Only valid if we rescale fibres
    if rescale_Y && normalize == :fibres
        # Compare:
        # If B_rescaled := avg_factor_sums * B,
        # Y_input ≈ A * B_rescaled
        #       Y ≈ A * B (Here, Y and B have normalized fibers)
        B_lateral_slices = eachslice(B, dims=2)
        B_lateral_slices .*= factor_sums
    end

    #@show kwargs

    return A, B, rel_errors, norm_grad, dist_Ncone
end

#----------------------- old rescaling functions -------------------------#

"""Rescales A and B so each factor (horizontal slices) of B has similar magnitude."""
function rescaleAB!(A, B; normalize, metricA, metricB, scaleBtoA)
    scaleBtoA ? nothing : throw(UnimplimentedError("rescaling from A to B is not implimented (YET!)"))

    if normalize == :fibres
        _avg_fibre_normalize!(A, B; metricB)
    elseif normalize == :slices
        _slice_normalize!(A, B; metricB)
    else
        return throw(UnimplimentedError("Other normalizations are not implimented (YET!)"))
    end
end

"""Rescales A and B so each factor (3 fibres) of B has similar magnitude."""
function _avg_fibre_normalize!(A::AbstractMatrix, B::AbstractArray; metricB)
    if metricB != :L1
        throw(UnimplimentedError("Other metrics are not implimented for this normalization (YET!)"))
    end

    fiber_sums = sum.(eachslice(B, dims=(1,2)))
    avg_factor_sums = mean.(eachrow(fiber_sums))

    B_horizontal_slices = eachslice(B, dims=1)
    B_horizontal_slices ./= avg_factor_sums

    A_rows = eachcol(A)
    A_rows .*= avg_factor_sums
end

"""Rescales A and B so each factor (horizontal slices) of B has similar magnitude."""
function _slice_normalize!(A::AbstractMatrix, B::AbstractArray; metricB) # B could be higher order
    B_horizontal_slices = eachslice(B, dims=1)

    fiber_sums = 1
    if metricB == :L1
        fiber_sums = sum.(B_horizontal_slices) #assumed positive
    elseif metricB == :Linfty
        fiber_sums = maximum.(B_horizontal_slices)
    else
        throw(UnimplimentedError("Other metrics are not implimented for this normalization (YET!)"))
    end

    B_horizontal_slices ./= fiber_sums

    A_rows = eachcol(A)
    A_rows .*= fiber_sums
end

function rescaleY(Y; normalize=:fibres, metric)
    if metric != :L1
        throw(UnimplimentedError("Other metrics are not implimented for rescaling Y (YET!)"))
    end

    if normalize == :fibres
        return _avg_fibre_rescale(Y)
    elseif normalize == :slices
        return _slice_rescale(Y)
    else
        return throw(UnimplimentedError("Other normalizations are not implimented (YET!)"))
    end
end

function _avg_fibre_rescale(Y)
    fiber_sums = sum.(eachslice(Y, dims=(1,2)))
    avg_fiber_sums = mean.(eachcol(fiber_sums))
    Yscaled = copy(Y)
    Y_lateral_slices = eachslice(Yscaled, dims=2)
    Y_lateral_slices ./= avg_fiber_sums
    return Yscaled, avg_fiber_sums
end

function _slice_rescale(Y)
    slice_sums = sum.(eachslice(Y, dims=1))
    Yscaled = copy(Y)
    Y ./= slice_sums
    return Yscaled, slice_sums
end

mean(x) = sum(x) / length(x)
