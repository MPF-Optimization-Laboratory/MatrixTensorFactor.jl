# Code for using the the new factorize internals with the old interface
#
# Wraps the new `factorize` function with nnmtf so that old code doesn't break,
# but can still benifit from the newer and more optimized internals.

# nnmtf can still call _nnmtf_proxgrad as before
# We just need to change the internals

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

    factorize_criterion = parse_criterion[criterion]
    constrain_output = false
    if allequal((metric,metricA,metricB))
        constrain_output = (metric == :L1)
    else
        @warn "(metric, metricA, metricB) = $((metric,metricA,metricB)) are not all the same, setting constrain_output=false"
    end
    decomposition = Tucker1(B, A)

    #--- output = factorize(input) ---#
    X, stats, _ = factorize(Y;
        model=Tucker1,
        decomposition,
        rank=R,
        stats=[Iteration, RelativeError, GradientNorm, GradientNNCone, ObjectiveValue],
        constraints=[constraintA, constraintB],
        converged=factorize_criterion,
        maxiter,
        tolerence=tol,
        momentum,
        δ=delta,
        constrain_output,
        constrain_init=false,
    )

    #--- Process output to return the same types as the old nntf ---#
    # Use collect to ensure they are all plain array types
    A = matrix_factor(X, 1) |> collect
    B = core(X) |> collect
    rel_errors = stats[:, :RelativeError] |> collect
    norm_grad = stats[:, :GradientNorm] |> collect
    dist_Ncone = stats[:, :GradientNNCone] |> collect

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

    return A, B, rel_errors, norm_grad, dist_Ncone
end

const parse_criterion = (
    :ncone => GradientNNCone,
    :iterates => Iteration,
    :objective => ObjectiveValue,
    :relativeerror => RelativeError,
)

const normalize_to_simplex_constraint = (
    :fibres => simplex_12slices!,
    :slices => simplex_1slices!,
    :rows => simplex_rows!,
    :cols => simplex_cols!,
)

const normalize_to_scaled_l1_constraint = (
    :fibres => l1scale_12slices! ∘ nonnegative!,
    :slices => l1scale_1slices! ∘ nonnegative!,
    :rows => l1scale_rows! ∘ nonnegative!,
    :cols => l1scale_cols! ∘ nonnegative!,
)

const normalize_to_scaled_linfty_constraint = (
    :fibres => linftyscale_12slices! ∘ nonnegative!,
    :slices => linftyscale_1slices! ∘ nonnegative!,
    :rows => linftyscale_rows! ∘ nonnegative!,
    :cols => linftyscale_cols! ∘ nonnegative!,
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

"""
    proj!(X::AbstractArray; projection=:nonnegative, dims=nothing)

Projects X according to projection.

When using the simplex projection, ensures each slice along dims is normalized.
"""
function proj!(X::AbstractArray; projection=:nonnegative, metric=:L1, dims=nothing)
    if projection == :nonnegative
        X .= ReLU.(X)

    elseif projection == :nnscale && (metric in (:L1, :Linfty))
        if isnothing(dims)
            throw(ArgumentError("normalize == :nothing and projection == :nnscale are uncompatible. Unsure what which part of X should be normalized."))
        else
            X_slices = eachslice(X; dims)
            for slice in X_slices
                # slices which contain exclusively nonpositive values should be projected using simplex
                # this ensures we don't project a slice to the origin, which cannot be normalized
                if all(x -> x <= 0, slice) #all(slice .<= 0)
                    if metric == :L1
                        slice .= projsplx(slice)
                    elseif metric == :Linfty
                        slice .= projmax!(slice)
                    end
                # otherwise only use ReLU and worry about normalization later
                else
                    slice .= ReLU.(slice)
                end
            end
        end

    elseif projection == :simplex && metric == :L1
        if isnothing(dims)
            throw(ArgumentError("normalize == :nothing and projection == :simplex are uncompatible. Unsure what which part of X should be projected to the simplex."))
        else
            X_slices = eachslice(X; dims)
            X_slices .= projsplx.(X_slices)
        end

    else
        throw(UnimplimentedError("The combination of projection $projection and metric $metric is not implemented (YET!)"))
    end
end

function projmax!(v) # TODO make this valid for any v, not just v where every element is negative
    v_max = maximum(v)
    v[v .< v_max] .= 0
    v[v .== v_max] .= 1
end
