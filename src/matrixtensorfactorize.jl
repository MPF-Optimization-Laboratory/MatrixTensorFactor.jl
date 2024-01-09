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
- `:nothing`: does not enforce any normalization of `B`
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
    IMPLIMENTED_CRITERIA::Set{Symbol}

- `:ncone`: vector-set distance between the -gradient of the objective and the normal cone
- `:iterates`: A,B before and after one iteration are close in L2 norm
- `:objective`: objective before and after one iteration is close
"""
const IMPLIMENTED_CRITERIA = Set{Symbol}((:ncone, :iterates, :objective))

"""
    IMPLIMENTED_STEPSIZES::Set{Symbol}

- `:lipshitz`: gradient step 1/L for lipshitz constant L
- `:spg`: spectral projected gradient stepsize
"""
const IMPLIMENTED_STEPSIZES = Set{Symbol}((:lipshitz, :spg))

"""Minimum step size allowed for spg stepsize method"""
const MIN_STEP = 1e-10
"""Maximum step size allowed for spg stepsize method"""
const MAX_STEP = 1e10

const IMPLIMENTED_OPTIONS = Dict(
    "normalizations" => IMPLIMENTED_NORMALIZATIONS,
    "projections" => IMPLIMENTED_PROJECTIONS,
    "criteria" => IMPLIMENTED_CRITERIA,
    "stepsizes" => IMPLIMENTED_STEPSIZES,
)

"""
    nnmtf(Y::Abstract3Tensor, R::Integer; kwargs...)

Non-negatively matrix-tensor factorizes an order 3 tensor Y with a given "rank" R.

Factorizes ``Y \\approx A B`` where ``\\displaystyle Y[i,j,k] \\approx \\sum_{r=1}^R A[i,r]*B[r,j,k]``
and the factors ``A, B \\geq 0`` are nonnegative.

Note there may NOT be a unique optimal solution

# Arguments
- `Y::Abstract3Tensor`: tensor to factorize
- `R::Integer`: rank to factorize Y (size(A)[2] and size(B)[1])

# Keywords
- `maxiter::Integer=100`: maxmimum number of iterations
- `tol::Real=1e-3`: desiered tolerance for the convergence criterion
- `rescale_AB::Bool=true`: scale B at each iteration so that the factors (horizontal slices) have similar 3-fiber sums.
- `rescale_Y::Bool=true`: Preprocesses the input `Y` to have normalized 3-fiber sums (on average), and rescales the final `B` so `Y=A*B`.
- `plot_B::Integer=0`: if not 0, plot B every plot_B iterations
- `names::AbstractVector{String}=String[]`: names of the slices of B to use for ploting
- `normalize::Symbol=:fibres`: part of B that should be normalized (must be in IMPLIMENTED_NORMALIZATIONS)
- `projection::Symbol=:nnscale`: constraint to use and method for enforcing it (must be in IMPLIMENTED_PROJECTIONS)
- `criterion::Symbol=:ncone`: how to determine if the algorithm has converged (must be in IMPLIMENTED_CRITERIA)
- `stepsize::Symbol=:lipshitz`: used for the gradient decent step (must be in IMPLIMENTED_STEPSIZES)

# Returns
- `A::Matrix{Float64}`: the matrix A in the factorization Y ≈ A * B
- `B::Array{Float64, 3}`: the tensor B in the factorization Y ≈ A * B
- `rel_errors::Vector{Float64}`: relative errors at each iteration
- `norm_grad::Vector{Float64}`: norm of the full gradient at each iteration
- `dist_Ncone::Vector{Float64}`: distance of the -gradient to the normal cone at each iteration
"""
function nnmtf(Y::Abstract3Tensor, R::Union{Nothing, Integer}=nothing;
    normalize::Symbol=:fibres,
    projection::Symbol=:nnscale,
    criterion::Symbol=:ncone,
    stepsize::Symbol=:lipshitz,
    kwargs...
)

    # Iteration option checking
    if !(projection in IMPLIMENTED_PROJECTIONS)
        return ArgumentError("projection is not an implimented projection")
    elseif !(normalize in IMPLIMENTED_NORMALIZATIONS)
        return ArgumentError("normalize is not an implimented normalization")
    elseif !(criterion in IMPLIMENTED_CRITERIA)
        return ArgumentError("criterion is not an implimented criterion")
    elseif !(stepsize in IMPLIMENTED_STEPSIZES)
        return ArgumentError("stepsize is not an implimented stepsize")
    end

    if isnothing(R) # TODO automatically estimate the rank
        # Run nnmtf with R from 1 to size(Y)[1]
        # Compare fit ||Y - AB||_F^2 across all R
        # Return the output at the maximum positive curavature of ||Y - AB||_F^2
        return UnimplimentedError("Rank Estimation not implimented (YET!)")
    end

    return _nnmtf_proxgrad(Y, R; normalize, projection, criterion, stepsize, kwargs...)
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
    plot_B::Integer=0,
    names::AbstractVector{String}=String[],
    normalize::Symbol=:fibres,
    projection::Symbol=:nonnegative,
    stepsize::Symbol=:lipshitz,
    criterion::Symbol=:ncone,
    rescale_AB::Bool = (projection == :nnscale ? true : false),
    rescale_Y::Bool = (projection == :nnscale ? true : false),
)
    # Override scaling if no normalization is requested
    normalize == :nothing ? (rescale_AB = rescale_Y = false) : nothing

    # Extract Dimentions
    M, N, P = size(Y)

    # Initialize A, B
    init(x...) = abs.(randn(x...))
    A = init(M, R)
    B = init(R, N, P)

    rescaleAB!(A, B; normalize)

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
    rel_errors[i] = residual(A*B, Y; normalize)
    grad_A, grad_B = calc_gradient(A, B, Y)
    norm_grad[i] = combined_norm(grad_A, grad_B)
    dist_Ncone[i] = dist_to_Ncone(grad_A, grad_B, A, B)

    A_last = copy(A)
    B_last = copy(B)

    step = nothing

    # Main Loop
    while i < maxiter
        A_last_last = copy(A_last) # for stepsizes that require the last two iterations
        B_last_last = copy(B_last)

        A_last = copy(A)
        B_last = copy(B)

        if (plot_B != 0) && ((i-1) % plot_B == 0)
            plot_factors(B, names, appendtitle=" at i=$i")
        end

        if i > 1 && stepsize == :spg
            grad_A_last_last, _ = calc_gradient(A_last_last, B_last_last, Y)
            grad_A_last, _ = calc_gradient(A_last, B_last, Y)
            step = spg_stepsize(A_last, A_last_last, grad_A_last, grad_A_last_last)
            #!isfinite(step) ? println(A_last, A_last_last, grad_A_last, grad_A_last_last) : nothing
        end

        grad_step_A!(A, B, Y; step)
        proj!(A; projection, dims=1) # Want the rows of A normalized when using :simplex projection

        if i > 1 && stepsize == :spg
            # note the mixed gradient below because A gets updated before B
            _, grad_B_last_last = calc_gradient(A_last, B_last_last, Y)
            _, grad_B_last = calc_gradient(A, B_last, Y)
            step = spg_stepsize(B_last, B_last_last, grad_B_last, grad_B_last_last)
        end

        grad_step_B!(A, B, Y; step)
        proj!(B; projection, dims=to_dims(normalize))

        rescale_AB ? rescaleAB!(A, B; normalize) : nothing

        # Calculate relative error and norm of gradient
        i += 1
        rel_errors[i] = residual(A*B, Y; normalize)
        grad_A, grad_B = calc_gradient(A, B, Y)
        norm_grad[i] = combined_norm(grad_A, grad_B)
        dist_Ncone[i] = dist_to_Ncone(grad_A, grad_B, A, B)

        if converged(; dist_Ncone, i, A, B, A_last, B_last, tol, problem_size, criterion, Y)
            break
        end
    end

    # Chop Excess
    keep_slice = 1:i
    rel_errors = rel_errors[keep_slice]
    norm_grad = norm_grad[keep_slice]
    dist_Ncone = dist_Ncone[keep_slice]

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

"""
Convergence criteria function.

When using :ncone, we "normalize" the distance vector so the tolerance can be picked
independent of the dimentions of Y and rank R.

Note the use of `;` in the function definition so that order of arguments does not matter,
and keyword assignment can be ignored if the input variables are named exactly as below.
"""
function converged(; dist_Ncone, i, A, B, A_last, B_last, tol, problem_size, criterion, Y)
    if !(criterion in IMPLIMENTED_CRITERIA)
        return UnimplimentedError("criterion is not an impliment criterion")
    elseif criterion == :ncone
        return dist_Ncone[i]/sqrt(problem_size) < tol
    elseif criterion == :iterates
        return combined_norm(A - A_last, B - B_last) < tol
    elseif criterion == :objective
        return 0.5 * norm(A*B - Y)^2 < tol
    end
end

"""
Converts the symbol normalize to the dimention(s) used to iterate or process the second
array B.
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
function spg_stepsize(x, x_last, grad_x, grad_x_last; min_step=MIN_STEP, max_step=MAX_STEP)
    s = x - x_last
    y = grad_x - grad_x_last
    step = sum(s[:] .^ 2) / sum(s[:] .* y[:])
    return max(min_step, min(step, max_step)) # safeguards to ensure step is within reasonable bounds
end

"""
    proj!(X::AbstractArray; projection=:nonnegative, dims=nothing)

Projects X according to projection.

When using the simplex projection, ensures each slice along dims is normalized.
"""
function proj!(X::AbstractArray; projection=:nonnegative, dims=nothing)
    if projection == :nonnegative
        X .= ReLU.(X)

    elseif projection == :nnscale
        if isnothing(dims)
            return ArgumentError("normalize == :nothing and projection == :nnscale are uncompatible. Unsure what which part of X should be normalized.")
        else
            X_slices = eachslice(X; dims)
            for slice in X_slices
                # slices which contain exclusively nonpositive values should be projected using simplex
                # this ensures we don't project a slice to the origin, which cannot be normalized
                if all(slice .<= 0)
                    slice .= projsplx(slice)
                # otherwise only use ReLU and worry about normalization later
                else
                    slice .= ReLU.(slice)
                end
            end
        end

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
    dist_to_Ncone(grad_A, grad_B, A, B)

Calculate the distance of the -gradient to the normal cone of the positive orthant.
"""
function dist_to_Ncone(grad_A, grad_B, A, B)
    grad_A_restricted = grad_A[(A .> 0) .|| (grad_A .< 0)]
    grad_B_restricted = grad_B[(B .> 0) .|| (grad_B .< 0)]
    return combined_norm(grad_A_restricted, grad_B_restricted)
end

# TODO move this ploting function to SedimentTools? Or seperate viz.jl file?
"""
    plot_factors(B, names; appendtitle="")

Plot each horizontal slice of B. Names give the name of each vertical slice.
"""
function plot_factors(B, names=string.(eachindex(B[1,:,1])); appendtitle="")
    size(B)[2] == length(names) || ArgumentError("names should have the same length as size(B)[2]")
    fiber_sums = sum.(eachslice(B,dims=(1,2)))
    avg_factor_sums = Diagonal(mean.(eachrow(fiber_sums)))
    B_temp = avg_factor_sums^(-1) * B
    for (j, B_slice) ∈ enumerate(eachslice(B_temp,dims=1))
        p = heatmap(B_slice,
            yticks=(eachindex(B_slice[:,1]), names),
            xticks=([1, length(B_slice[1,:])],["0", "1"]),
            xlabel="Normalized Range of Values",
            title = "Learned Distributions for Factor $j" * appendtitle,
        )
        display(p)
    end
end

function grad_step_A!(A, B, Y; step=nothing)
    @einsum BB[s,r] := B[s,j,k]*B[r,j,k]
    @einsum GG[i,r] := Y[i,j,k]*B[r,j,k]
    grad = A*BB .- GG
    isnothing(step) ? step = 1/norm(BB) : nothing # Lipshitz fallback
    A .-= step .* grad # gradient step
end

function grad_step_B!(A, B, Y; step=nothing)
    AA = A'A
    grad = AA*B .- A'*Y
    isnothing(step) ? step = 1/norm(AA) : nothing # Lipshitz fallback
    B .-= step .* grad # gradient step
end

function calc_gradient(A, B, Y)
    @einsum BB[s,r] := B[s,j,k]*B[r,j,k]
    @einsum GG[i,r] := Y[i,j,k]*B[r,j,k]
    AA = A'A
    grad_A = A*BB .- GG
    grad_B = AA*B .- A'*Y
    return grad_A, grad_B
end

# Could compute the gradients this way to reuse CF-Y,
# but the first way is still faster!
#=
CFY = A*B .- Y
@einsum grad_A[i,r] := CFY[i,j,k]*B[r,j,k]
@einsum grad_B[r,j,k] := A[i,r]*CFY[i,j,k]
return grad_A, grad_B
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
#    A = sum.(eachslice(B, dims=(1,2)))
#    B ./= A
#    A .= A * A
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
