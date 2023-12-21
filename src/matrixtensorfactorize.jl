#=
Holds the block coordinate decent factorization function
and related helpers
=#

struct UnimplimentedError <: Exception end

"""
    IMPLIMENTED_NORMALIZATIONS::Set{Symbol}

- `:fibre`: set ``\\sum_{k=1}^K B[r,j,k] = 1`` for all ``r, j``, or when `projection==:nnscale`,
    set ``\\sum_{j=1}^J\\sum_{k=1}^K B[r,j,k] = J`` for all ``r``
- `:slice`: set ``\\sum_{j=1}^J\\sum_{k=1}^K B[r,j,k] = 1`` for all ``r``
- `:nothing`: does not enforce any normalization of `F`
"""
const IMPLIMENTED_NORMALIZATIONS = Set{Symbol}((:fibres, :slices, :nothing))

"""
    IMPLIMENTED_PROJECTIONS::Set{Symbol}

- `:nnscale`: Two stage block coordinate decent; 1) projected gradient decent onto nonnegative
    orthant, 2) shift any weight from `B` to `A` according to normalization. Equivilent to
    :nonnegative when `normalization==:nothing`.
"""
const IMPLIMENTED_PROJECTIONS = Set{Symbol}((:nnscale, :simplex, :nonnegative)) # nn is nonnegative

"""
    IMPLIMENTED_CRITERIONS::Set{Symbol}

- `:ncone`: vector-set distance between the -gradient of the objective and the normal cone
- `:iterates`: A,B before and after one iteration are close in L2 norm
- `:objective`: objective before and after one iteration is close
"""
const IMPLIMENTED_CRITERIONS = Set{Symbol}((:ncone, :iterates, :objective))

"""
    IMPLIMENTED_STEPSIZES::Set{Symbol}

- `:lipshitz`: gradient step 1/L for lipshitz constant L
- `:spg`: spectral projected gradient stepsize
"""
const IMPLIMENTED_STEPSIZES = Set{Symbol}((:lipshitz, :spg))

"""
    nnmtf(Y::Abstract3Tensor, R::Integer; kwargs...)

Non-negatively matrix-tensor factorizes an order 3 tensor Y with a given "rank" R.

Factorizes ``Y \\approx C F`` where ``\\displaystyle Y[i,j,k] \\approx \\sum_{r=1}^R C[i,r]*F[r,j,k]``
and the factors ``C, F \\geq 0`` are nonnegative.

Note there may NOT be a unique optimal solution

# Arguments
- `Y::Abstract3Tensor`: tensor to factorize
- `R::Integer`: rank to factorize Y (size(C)[2] and size(F)[1])

# Keywords
- `maxiter::Integer=100`: maxmimum number of iterations
- `tol::Real=1e-3`: desiered tolerance for the convergence criterion
- `rescale_CF::Bool=true`: scale F at each iteration so that the factors (horizontal slices) have similar 3-fiber sums.
- `rescale_Y::Bool=true`: Preprocesses the input `Y` to have normalized 3-fiber sums (on average), and rescales the final `F` so `Y=C*F`.
- `plot_F::Integer=0`: if not 0, plot F every plot_F iterations
- `names::AbstractVector{String}=String[]`: names of the slices of F to use for ploting
- `normalize::Symbol=:fibres`: part of F that should be normalized (must be in IMPLIMENTED_NORMALIZATIONS)
- `projection::Symbol=:nnscale`: constraint to use and method for enforcing it (must be in IMPLIMENTED_PROJECTIONS)
- `criterion::Symbol=:ncone`: how to determine if the algorithm has converged (must be in IMPLIMENTED_CRITERIONS)

# Returns
- `C::Matrix{Float64}`: the matrix C in the factorization Y ≈ C * F
- `F::Array{Float64, 3}`: the tensor F in the factorization Y ≈ C * F
- `rel_errors::Vector{Float64}`: relative errors at each iteration
- `norm_grad::Vector{Float64}`: norm of the full gradient at each iteration
- `dist_Ncone::Vector{Float64}`: distance of the -gradient to the normal cone at each iteration
"""
function nnmtf(Y::Abstract3Tensor, R::Union{Nothing, Integer}=nothing;
    normalize=:fibres,
    projection::Symbol=:nnscale,
    kwargs...
)
    if isnothing(R)
        # Run nnmtf with R from 1 to size(Y)[1]
        # Compare fit ||Y - AB||_F^2 across all R
        # Return the output at the maximum positive curavature of ||Y - AB||_F^2
        return UnimplimentedError("Rank Estimation not implimented (YET!)")
    end

    if !(projection in IMPLIMENTED_PROJECTIONS)
        return ArgumentError("projection is not an implimented projection")
    elseif !(normalize in IMPLIMENTED_NORMALIZATIONS)
            return ArgumentError("normalize is not an implimented normalization")
    else
        return _nnmtf_proxgrad(Y, R; normalize, projection, kwargs...)
    end
end

"""
nnmtf using proximal (projected) gradient decent alternating through blocks (BCD)

Explination of argument handeling:
- if normalize == :nothing, we do not want any rescaling at all
- if projection == :nnscale, we should by default, rescale the factors
- if projection is anything else, we should not, by default, rescale the factors
"""
function _nnmtf_proxgrad(
    Y::Abstract3Tensor,
    R::Integer;
    maxiter::Integer=1000,
    tol::Real=1e-4,
    plot_F::Integer=0,
    names::AbstractVector{String}=String[],
    normalize::Symbol=:fibres,
    projection::Symbol=:nonnegative,
    criterion::Symbol=:ncone,
    rescale_CF::Bool = (projection == :nnscale ? true : false),
    rescale_Y::Bool = (projection == :nnscale ? true : false),
    kwargs...
)
    # Override scaling if no normalization is requested
    normalize == :nothing ? (rescale_CF = rescale_Y = false) : nothing

    # Extract Dimentions
    M, N, P = size(Y)

    # Initialize C, F
    init(x...) = abs.(randn(x...))
    C = init(M, R)
    F = init(R, N, P)

    rescaleAB!(C, F; normalize)

    problem_size = R*(M + N*P)

    # # Scale Y if desired
    if rescale_Y
        # Y_input = copy(Y)
        Y, factor_sums = rescaleY(Y; normalize)
    end

    # Initialize Looping
    i = 1
    rel_errors = zeros(maxiter)
    norm_grad = zeros(maxiter)
    dist_Ncone = zeros(maxiter)

    # Calculate initial relative error and gradient
    rel_errors[i] = residual(C*F, Y; normalize)
    grad_C, grad_F = calc_gradient(C, F, Y)
    norm_grad[i] = combined_norm(grad_C, grad_F)
    dist_Ncone[i] = dist_to_Ncone(grad_C, grad_F, C, F)

    C_last = copy(C)
    F_last = copy(F)
    # Main Loop
    # Ensure at least 1 step is performed
    while (i == 1) || (!converged(; dist_Ncone, i, C, F, C_last, F_last, tol, problem_size, criterion) && (i < maxiter))
        C_last = copy(C)
        F_last = copy(F)

        if (plot_F != 0) && ((i-1) % plot_F == 0)
            plot_factors(F, names, appendtitle=" at i=$i")
        end

        grad_step_C!(C, F, Y)
        proj!(C; projection, dims=1) # Always want the rows of C normalized regardless of how F is normalized
        grad_step_F!(C, F, Y)
        proj!(F; projection, dims=to_dims(normalize))

        rescale_CF ? rescaleAB!(C, F; normalize) : nothing

        # Calculate relative error and norm of gradient
        i += 1
        rel_errors[i] = residual(C*F, Y; normalize)
        grad_C, grad_F = calc_gradient(C, F, Y)
        norm_grad[i] = combined_norm(grad_C, grad_F)
        dist_Ncone[i] = dist_to_Ncone(grad_C, grad_F, C, F)
    end

    # Chop Excess
    keep_slice = 1:i
    rel_errors = rel_errors[keep_slice]
    norm_grad = norm_grad[keep_slice]
    dist_Ncone = dist_Ncone[keep_slice]

    # Rescale F back if Y was initialy scaled
    # Only valid if we rescale fibres
    if rescale_Y && normalize == :fibres
        # Compare:
        # If F_rescaled := avg_factor_sums * F,
        # Y_input ≈ C * F_rescaled
        #       Y ≈ C * F (Here, Y and F have normalized fibers)
        F_lateral_slices = eachslice(F, dims=2)
        F_lateral_slices .*= factor_sums
    end

    return C, F, rel_errors, norm_grad, dist_Ncone
end

# Convergence criteria. We "normalize" the distance vector so the tolerance can be
# picked independent of the dimentions of Y and rank R
function converged(; dist_Ncone, i, C, F, C_last, F_last, tol, problem_size, criterion)
    if !(criterion in IMPLIMENTED_CRITERIONS)
        return UnimplimentedError("criterion is not an impliment criterion")
    elseif criterion == :ncone
        return dist_Ncone[i]/sqrt(problem_size) < tol
    elseif criterion == :iterates
        return combined_norm(C - C_last, F - F_last) < tol
    elseif criterion == :objective
        return rel_errors[i] < tol
    end
end

"""
Converts the symbol normalize to the dimention(s) used to iterate or process the second
array F.
"""
function to_dims(normalize::Symbol)
    if normalize == :fibres
        return (1, 2)
    elseif normalize == :slices
        return 1
    elseif normalize == :nothing
        return nothing
    else
        return UnimplimentedError("normalize is not an implimented normalization")
    end
end

"""
    residual(Yhat, Y; normalize=:nothing)

Wrapper to use the relative error calculation according to the normalization used.

- `normalize==:nothing`: entry-wise L2 relative error between the two arrays
- `normalize==:fibres`: average L2 relative error between all 3-fibres
- `normalize==:slices`: average L2 relative error between all 1-mode slices

See also [`rel_error`](@ref), [`mean_rel_error`](@ref).
"""
function residual(Yhat, Y; normalize=:nothing)
    if normalize in (:fibres, :slices)
        return mean_rel_error(Yhat, Y; dims=to_dims(normalize))
    elseif normalize == :nothing
        return rel_error(Yhat, Y)
    else
        return UnimplimentedError("normalize is not an implimented normalization")
    end
end

"""
spectral projected gradient stepsize
"""
function spg_stepsize(x, x_last, grad_x, grad_x_last)
    s = x - x_last
    y = grad_x - grad_x_last
    return sum(s[:] .^ 2) / sum(s[:] .* y[:])
end

"""
    proj!(X::AbstractArray; projection=:nonnegative, dims=nothing)

Projects X according to projection.

When using the simplex projection, ensures each slice along dims is normalized.
"""
function proj!(X::AbstractArray; projection=:nonnegative, dims=nothing)
    if projection in (:nonnegative, :nnscale)
        X .= ReLU.(X)

    elseif projection == :simplex
        if isnothing(dims)
            return ArgumentError("normalize == :nothing and projection == :simplex are uncompatible. Unsure what which part of X should be projected to the simplex.")
        else
            X_slices = eachslice(X; dims)
            X_slices .= projsplx.(X_slices)
        end
    end
end

"""
    dist_to_Ncone(grad_C, grad_F, C, F)

Calculate the distance of the -gradient to the normal cone of the positive orthant.
"""
function dist_to_Ncone(grad_C, grad_F, C, F)
    grad_C_restricted = grad_C[(C .> 0) .|| (grad_C .< 0)]
    grad_F_restricted = grad_F[(F .> 0) .|| (grad_F .< 0)]
    return combined_norm(grad_C_restricted, grad_F_restricted)
end

# TODO move this ploting function to SedimentTools? Or seperate viz.jl file?
"""
    plot_factors(F, names; appendtitle="")

Plot each horizontal slice of F. Names give the name of each vertical slice.
"""
function plot_factors(F, names=string.(eachindex(F[1,:,1])); appendtitle="")
    size(F)[2] == length(names) || ArgumentError("names should have the same length as size(F)[2]")
    fiber_sums = sum.(eachslice(F,dims=(1,2)))
    avg_factor_sums = Diagonal(mean.(eachrow(fiber_sums)))
    F_temp = avg_factor_sums^(-1) * F
    for (j, F_slice) ∈ enumerate(eachslice(F_temp,dims=1))
        p = heatmap(F_slice,
            yticks=(eachindex(F_slice[:,1]), names),
            xticks=([1, length(F_slice[1,:])],["0", "1"]),
            xlabel="Normalized Range of Values",
            title = "Learned Distributions for Factor $j" * appendtitle,
        )
        display(p)
    end
end

function grad_step_C!(C, F, Y; step=nothing)
    @einsum FF[s,r] := F[s,j,k]*F[r,j,k]
    @einsum GG[i,r] := Y[i,j,k]*F[r,j,k]
    isnothing(step) ? step = 1/norm(FF) : nothing # Lipshitz fallback
    grad = C*FF .- GG
    C .-= step .* grad # gradient step
end

function grad_step_F!(C, F, Y; step=nothing)
    CC = C'C
    isnothing(step) ? step = 1/norm(CC) : nothing # Lipshitz fallback
    grad = CC*F .- C'*Y
    F .-= step .* grad # gradient step
    #F .= ReLU.(F) # project
end

function calc_gradient(C, F, Y)
    @einsum FF[s,r] := F[s,j,k]*F[r,j,k]
    @einsum GG[i,r] := Y[i,j,k]*F[r,j,k]
    CC = C'C
    grad_C = C*FF .- GG
    grad_F = CC*F .- C'*Y
    return grad_C, grad_F
end

# Could compute the gradients this way to reuse CF-Y,
# but the first way is still faster!
#=
CFY = C*F .- Y
@einsum grad_C[i,r] := CFY[i,j,k]*F[r,j,k]
@einsum grad_F[r,j,k] := C[i,r]*CFY[i,j,k]
return grad_C, grad_F
=#

"""Rescales A and B so each factor (horizontal slices) of B has similar magnitude."""
function rescaleAB!(A, B; normalize)
    if normalize == :fibres
        _avg_fibre_normalize!(A, B)
    elseif normalize == :slices
        _slice_normalize!(A, B)
    else
        return UnimplimentedException("Other normalizations are not implimented (YET!)")
    end
end

"""Rescales A and B so each factor (3 fibres) of B has similar magnitude."""
function _avg_fibre_normalize!(A::AbstractMatrix, B::Abstract3Tensor)
    fiber_sums = sum.(eachslice(B, dims=(1,2)))
    avg_factor_sums = mean.(eachrow(fiber_sums))

    B_horizontal_slices = eachslice(B, dims=1)
    B_horizontal_slices ./= avg_factor_sums

    A_rows = eachcol(A)
    A_rows .*= avg_factor_sums
end

"""Rescales A and B so each factor (horizontal slices) of B has similar magnitude."""
function _slice_normalize!(A::AbstractMatrix, B::AbstractArray) # B could be higher order
    fiber_sums = sum.(eachslice(B, dims=1))
    #avg_factor_sums = mean.(eachrow(fiber_sums))

    B_horizontal_slices = eachslice(B, dims=1)
    B_horizontal_slices ./= fiber_sums

    A_rows = eachcol(A)
    A_rows .*= fiber_sums
end

#function rescale2CF!(A, B)
#    C = sum.(eachslice(B, dims=(1,2)))
#    B ./= C
#    A .= A * C
#end

function rescaleY(Y; normalize=:fibres)
    if normalize == :fibres
        return _avg_fibre_rescale(Y)
    elseif normalize == :slices
        return _slice_rescale(Y)
    else
        return UnimplimentedException("Other normalizations are not implimented (YET!)")
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
