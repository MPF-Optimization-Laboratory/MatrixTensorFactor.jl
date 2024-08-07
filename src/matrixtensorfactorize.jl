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

"""
    MIN_STEP = 1e-10

Minimum step size allowed for spg stepsize method.
"""
const MIN_STEP = 1e-10

"""
    MAX_STEP = 1e10

Maximum step size allowed for spg stepsize method.
"""
const MAX_STEP = 1e10

"""Lists all implimented options"""
const IMPLIMENTED_OPTIONS = Dict(
    "normalizations" => IMPLIMENTED_NORMALIZATIONS,
    "projections" => IMPLIMENTED_PROJECTIONS,
    "criteria" => IMPLIMENTED_CRITERIA,
    "stepsizes" => IMPLIMENTED_STEPSIZES,
)

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
- `projection::Symbol=:nnscale`: constraint to use and method for enforcing it (must be in IMPLIMENTED_PROJECTIONS)
- `criterion::Symbol=:ncone`: how to determine if the algorithm has converged (must be in IMPLIMENTED_CRITERIA)
- `stepsize::Symbol=:lipshitz`: used for the gradient decent step (must be in IMPLIMENTED_STEPSIZES)
- `momentum::Bool=false`: use momentum updates
- `delta::Real=0.9999`: safeguard for maximum amount of momentum (see eq (3.5) Xu & Yin 2013)
- `R_max::Integer=size(Y)[1]`: maximum rank to try if R is not given
- `projectionA::Symbol=projection`: projection to use on factor A (must be in IMPLIMENTED_PROJECTIONS)
- `projectionB::Symbol=projection`: projection to use on factor B (must be in IMPLIMENTED_PROJECTIONS)
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

"""
nnmtf using proximal (projected) gradient decent alternating through blocks (BCD)

Explination of argument handeling:
- if normalize == :nothing, we do not want any rescaling at all
- if projection == :nnscale, we should by default, rescale the factors
- if projection is anything else, we should not, by default, rescale the factors
"""
function _nnmtf_proxgrad(
    Y::AbstractArray,
    R::Integer;
    maxiter::Integer=1000,
    tol::Real=1e-4,
    normalize::Symbol=:fibres,
    projection::Symbol=:nonnegative,
    stepsize::Symbol=:lipshitz,
    criterion::Symbol=:ncone,
    momentum::Bool=false,
    delta::Real=0.9999,
    rescale_AB::Bool = (projection == :nnscale ? true : false),
    rescale_Y::Bool = (projection == :nnscale ? true : false),
    projectionA::Symbol = projection,
    projectionB::Symbol = projection,
    A_init::Union{Nothing, AbstractMatrix}=nothing,
    B_init::Union{Nothing, AbstractArray}=nothing,
)
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

    if A_init === nothing
        B = _init(R, Ns...)
    else
        size(B_init) == (R, Ns...) || throw(ArgumentError("A_init should have size $((R, Ns...)), got $(size(B_init))"))
        B = B_init
    end

    # Only want to rescale the initialization if both A and B were not given
    # Otherwise, we should use the provided initialization
    if rescale_AB && A_init === nothing && B_init === nothing
        rescaleAB!(A, B; normalize)
    end

    problem_size = R*(M + prod(Ns))

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
    Yhat = A*B
    rel_errors[i] = relative_error(Yhat, Y; normalize)
    grad_A, grad_B = calc_gradient(A, B, Y)
    norm_grad[i] = combined_norm(grad_A, grad_B)
    dist_Ncone[i] = dist_to_Ncone(grad_A, grad_B, A, B)

    A_last = copy(A)
    B_last = copy(B)

    A_last_last = copy(A_last) # for stepsizes that require the last two iterations
    B_last_last = copy(B_last)

    step = nothing

    # Momentum variables
    t = 1
    LA = lipshitzA(B)
    LB = lipshitzB(A)
    LA_last = LA
    LB_last = LB

    # Main Loop
    while i < maxiter

        if momentum
            t_last = t
            t = 0.5*(1 + sqrt(1 + 4*t_last^2))
            omegahat = (t_last - 1) / t # Candidate momentum step
            LA = lipshitzA(B)
            omegaA = min(omegahat, delta*sqrt(LA_last/LA)) # Safeguarded momentum step
            #A = A_last + omegaA * (A_last - A_last_last)
            @. A += omegaA * (A_last - A_last_last)
        end

        if i > 1 && stepsize == :spg
            grad_A_last_last = calc_gradientA(A_last_last, B_last_last, Y)
            grad_A_last = calc_gradientA(A_last, B_last, Y)
            step = spg_stepsize(A_last, A_last_last, grad_A_last, grad_A_last_last)
        end

        grad_step_A!(A, B, Y; step)
        proj!(A; projection=projectionA, dims=1) # Want the rows of A normalized when using :simplex projection

        if momentum
            LB = lipshitzB(A)
            omegaB = min(omegahat, delta*sqrt(LB_last/LB))
            #B = B_last + omegaB * (B_last - B_last_last)
            @. B += omegaB * (B_last - B_last_last)
        end

        if i > 1 && stepsize == :spg
            # note the mixed gradient below (A_last with B_last_last) because A gets updated before B
            grad_B_last_last = calc_gradientB(A_last, B_last_last, Y)
            grad_B_last = calc_gradientB(A, B_last, Y)
            step = spg_stepsize(B_last, B_last_last, grad_B_last, grad_B_last_last)
        end

        grad_step_B!(A, B, Y; step)
        proj!(B; projection=projectionB, dims=to_dims(normalize))

        rescale_AB ? rescaleAB!(A, B; normalize) : nothing

        # Calculate relative error and norm of gradient
        i += 1
        Yhat .= A*B
        rel_errors[i] = relative_error(Yhat, Y; normalize)
#        grad_A, grad_B = calc_gradient(A, B, Y)
        grad_A .= calc_gradientA(A, B, Y)
        grad_B .= calc_gradientB(A, B, Y)
        norm_grad[i] = combined_norm(grad_A, grad_B)
        #        norm_grad[i] = combined_norm(grad_A, grad_B)
        dist_Ncone[i] = dist_to_Ncone(grad_A, grad_B, A, B)

        if converged(; dist_Ncone, i, A, B, A_last, B_last, tol, problem_size, criterion, Y, Yhat, normalize)
            break
        end

        A_last_last .= A_last #copy(A_last) # for stepsizes that require the last two iterations
        B_last_last .= B_last #copy(B_last)

        A_last .= A #copy(A)
        B_last .= B #copy(B)

        LA_last = LA
        LB_last = LB
    end

    # Chop Excess
    keep_slice = 1:i
    rel_errors = rel_errors[keep_slice]
    norm_grad = norm_grad[keep_slice]
    dist_Ncone = dist_Ncone[keep_slice]

    # If using nnscale, A and B may only be aproximatly normalized. So we need to project A
    # and B to the simplex to ensure they are exactly normalized.
    if projection == :nnscale
        proj!(A; projection=:simplex, dims=1)
        proj!(B; projection=:simplex, dims=to_dims(normalize))
    end

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
Default initialization
"""
_init(x...) = abs.(randn(x...))

"""
Convergence criteria function.

When using :ncone, we "normalize" the distance vector so the tolerance can be picked
independent of the dimentions of Y and rank R.

Note the use of `;` in the function definition so that order of arguments does not matter,
and keyword assignment can be ignored if the input variables are named exactly as below.
"""
function converged(; dist_Ncone, i, A, B, A_last, B_last, tol, problem_size, criterion, Y, Yhat, normalize)
    criterion_value = 0.0

    if !(criterion in IMPLIMENTED_CRITERIA)
        return UnimplimentedError("criterion is not an impliment criterion")

    elseif criterion == :ncone
        criterion_value = dist_Ncone[i]/sqrt(problem_size) #TODO remove root problem size dependence

    elseif criterion == :iterates
        criterion_value = combined_norm(A - A_last, B - B_last)

    elseif criterion == :objective
        criterion_value = 0.5 * norm(Yhat - Y)^2

    elseif criterion == :relativeerror
        criterion_value = relative_error(Yhat, Y; normalize)
    end

    return criterion_value < tol
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
    relative_error(Yhat, Y; normalize=:nothing)

Wrapper to use the relative error calculation according to the normalization used.

- `normalize==:nothing`: entry-wise L2 relative error between the two arrays
- `normalize==:fibres`: average L2 relative error between all 3-fibres
- `normalize==:slices`: average L2 relative error between all 1-mode slices

See also [`rel_error`](@ref), [`mean_rel_error`](@ref).
"""
function relative_error(Yhat, Y; normalize=:nothing)
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
            throw(ArgumentError("normalize == :nothing and projection == :nnscale are uncompatible. Unsure what which part of X should be normalized."))
        else
            X_slices = eachslice(X; dims)
            for slice in X_slices
                # slices which contain exclusively nonpositive values should be projected using simplex
                # this ensures we don't project a slice to the origin, which cannot be normalized
                if all(x -> x <= 0, slice) #all(slice .<= 0)
                    slice .= projsplx(slice)
                # otherwise only use ReLU and worry about normalization later
                else
                    slice .= ReLU.(slice)
                end
            end
        end

    elseif projection == :simplex
        if isnothing(dims)
            throw(ArgumentError("normalize == :nothing and projection == :simplex are uncompatible. Unsure what which part of X should be projected to the simplex."))
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
    grad_A_restricted = grad_A[@. (A > 0) | (grad_A < 0)]
    grad_B_restricted = grad_B[@. (B > 0) | (grad_B < 0)]
    return combined_norm(grad_A_restricted, grad_B_restricted)
end

function lipshitzA(B)
    BB = slicewise_dot(B, B) #@einsum BB[s,r] := B[s,j,k]*B[r,j,k]
    return opnorm(BB)
end

function lipshitzB(A)
    AA = A'A
    return opnorm(AA)
end

function grad_step_A!(A, B, Y; step=nothing) # not using calc_gradientA and lipshitzA so that BB only needs to be computed once
    BB = slicewise_dot(B, B) #@einsum BB[s,r] := B[s,j,k]*B[r,j,k]
    YB = slicewise_dot(Y, B) #@einsum GG[i,r] := Y[i,j,k]*B[r,j,k]
    grad = A*BB - YB
    isnothing(step) ? step = 1/opnorm(BB) : nothing # Lipshitz fallback
    @. A -= step * grad # gradient step
end

function grad_step_B!(A, B, Y; step=nothing)
    AA = A'A
    grad = AA*B - A'*Y
    isnothing(step) ? step = 1/opnorm(AA) : nothing # Lipshitz fallback
    @. B -= step * grad # gradient step
end

function calc_gradient(A, B, Y)
    return calc_gradientA(A, B, Y), calc_gradientB(A, B, Y)
end

function calc_gradientA(A, B, Y)
    BB = slicewise_dot(B, B) #@einsum BB[s,r] := B[s,j,k]*B[r,j,k]
    YB = slicewise_dot(Y, B) #@einsum GG[i,r] := Y[i,j,k]*B[r,j,k]
    grad_A = A*BB - YB
    return grad_A
end

function calc_gradientB(A, B, Y)
    AA = A'A
    grad_B = AA*B - A'*Y
    return grad_B
end

# Could compute the gradients this way to reuse CF-Y,
# but the first way is still faster!
#=
CFY = A*B - Y
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
function _avg_fibre_normalize!(A::AbstractMatrix, B::AbstractArray)
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


"""
nnmtf using Online proximal (projected) gradient decent alternating through blocks (BCD)

Updates Y each iteration with a new sample

"""
function nnmtf_proxgrad_online(
    Y::AbstractArray{T, 3},
    R::Integer,
    distributions,
    number_of_samples;
    xs,
    ys,
    maxiter::Integer=1000,
    tol::Real=1e-4,
    normalize::Symbol=:fibres,
    projection::Symbol=:nonnegative,
    stepsize::Symbol=:lipshitz,
    criterion::Symbol=:ncone,
    momentum::Bool=false,
    delta::Real=0.9999,
    rescale_AB::Bool = (projection == :nnscale ? true : false),
    rescale_Y::Bool = (projection == :nnscale ? true : false),
) where T <: Real
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
    Yhat = A*B
    rel_errors[i] = relative_error(Yhat, Y; normalize)
    grad_A, grad_B = calc_gradient(A, B, Y)
    norm_grad[i] = combined_norm(grad_A, grad_B)
    dist_Ncone[i] = dist_to_Ncone(grad_A, grad_B, A, B)

    A_last = copy(A)
    B_last = copy(B)

    step = nothing

    # Momentum variables
    t = 1
    LA = lipshitzA(B)
    LB = lipshitzB(A)

    # Main Loop
    while i < maxiter
        A_last_last = copy(A_last) # for stepsizes that require the last two iterations
        B_last_last = copy(B_last)

        A_last = copy(A)
        B_last = copy(B)

        LA_last = LA
        LB_last = LB

        if momentum
            t_last = t
            t = 0.5*(1 + sqrt(1 + 4*t_last^2))
            omegahat = (t_last - 1) / t # Candidate momentum step
            LA = lipshitzA(B)
            omegaA = min(omegahat, delta*sqrt(LA_last/LA)) # Safeguarded momentum step
            A = A_last + omegaA * (A_last - A_last_last)
        end

        if i > 1 && stepsize == :spg
            grad_A_last_last = calc_gradientA(A_last_last, B_last_last, Y)
            grad_A_last = calc_gradientA(A_last, B_last, Y)
            step = spg_stepsize(A_last, A_last_last, grad_A_last, grad_A_last_last)
        end

        grad_step_A!(A, B, Y; step)
        proj!(A; projection, dims=1) # Want the rows of A normalized when using :simplex projection

        if momentum
            LB = lipshitzB(A)
            omegaB = min(omegahat, delta*sqrt(LB_last/LB))
            B = B_last + omegaB * (B_last - B_last_last)
        end

        if i > 1 && stepsize == :spg
            # note the mixed gradient below (A_last with B_last_last) because A gets updated before B
            grad_B_last_last = calc_gradientB(A_last, B_last_last, Y)
            grad_B_last = calc_gradientB(A, B_last, Y)
            step = spg_stepsize(B_last, B_last_last, grad_B_last, grad_B_last_last)
        end

        grad_step_B!(A, B, Y; step)
        proj!(B; projection, dims=to_dims(normalize))

        rescale_AB ? rescaleAB!(A, B; normalize) : nothing

        # Calculate relative error and norm of gradient
        i += 1
        Yhat .= A*B
        rel_errors[i] = relative_error(Yhat, Y; normalize)
        grad_A, grad_B = calc_gradient(A, B, Y)
        norm_grad[i] = combined_norm(grad_A, grad_B)
        dist_Ncone[i] = dist_to_Ncone(grad_A, grad_B, A, B)

        if converged(; dist_Ncone, i, A, B, A_last, B_last, tol, problem_size, criterion, Y, Yhat, normalize)
            break
        end

        # Update Y using a new sample from distributions

        for i in 1:M
            new_sample = rand(distributions[i])
            updateY!(Y, new_sample; N=number_of_samples, I=i, xs, ys)
        end
        number_of_samples += 1
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

function updateY!(Y, new_sample; N, I, xs, ys)
    # Turn new_sample point into a gaussian over xs, ys
    a, b = new_sample
    σ = N^(-0.2) #bandwidth should get small as N gets big
    kernel = @. (2π*σ^2)^(-1)*exp(-0.5*((xs'-a)^2 + (ys-b)^2)/σ^2)
    kernel ./= sum(kernel)

    Y[I, :, :] .= (Y[I, :, :] .* N .+ kernel) ./ (N+1)
end
