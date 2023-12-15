#=
Holds the block coordinate decent factorization function
and related helpers

Temporarily make a separate BCD function so I can tweak it independently of the original
code before merging
# TODO merge code

=#

"""
    nnmtf2d(Y::Abstract3Tensor, R::Integer; kwargs...)

Non-negatively matrix-tensor factorizes an order 3 tensor Y with a given "rank" R.

Similar to [`nnmtf`](@ref) but the renormalization occurs on the horizontal slices of
the tensors Y and F, rather than the 3-fibre sums. That is,
- sum(Y[i,:,:]) is 1 for all i
- sum(F[r,:,:]) is 1 for all r
rather than sum(Y[i,j,:]) being 1 for all i and j.
"""
function nnmtf2d(
    Y::Abstract3Tensor,
    R::Integer;
    maxiter::Integer=1000,
    tol::Real=1e-4,
    rescale_Y::Bool=true,
    rescale_CF::Bool=true,
    plot_F::Integer=0,
    names::AbstractVector{String}=String[],
)
    # Extract Dimentions
    M, N, P = size(Y)

    # Initialize C, F
    init(x...) = abs.(randn(x...))
    C = init(M, R)
    F = init(R, N, P)

    rescaleCF2!(C, F)

    problem_size = R*(M + N*P)

    # Scale Y if desired
    if rescale_Y
        # Y_input = copy(Y)
        Y, slice_sums = rescaleY2(Y)
    end

    # Initialize Looping
    i = 1
    rel_errors = zeros(maxiter)
    norm_grad = zeros(maxiter)
    dist_Ncone = zeros(maxiter)

    # Calculate initial relative error and gradient
    rel_errors[i] = mean_rel_error(Y, C*F, dims=1)
    grad_C, grad_F = calc_gradient(C, F, Y)
    norm_grad[i] = combined_norm(grad_C, grad_F)
    dist_Ncone[i] = dist_to_Ncone(grad_C, grad_F, C, F)

    # Convergence criteria. We "normalize" the distance vector so the tolerance can be
    # picked independent of the dimentions of Y and rank R
    converged(dist_Ncone, i) = dist_Ncone[i]/sqrt(problem_size) < tol

    # Main Loop
    # Ensure at least 1 step is performed
    while (i == 1) || (!converged(dist_Ncone, i) && (i < maxiter))
        if (plot_F != 0) && ((i-1) % plot_F == 0)
            plot_factors(F, names, appendtitle=" at i=$i")
        end

        updateC!(C, F, Y)
        updateF!(C, F, Y)

        rescale_CF ? rescaleCF2!(C, F) : nothing

        # Calculate relative error and norm of gradient
        i += 1
        rel_errors[i] = mean_rel_error(C*F, Y, dims=1)
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
    if rescale_Y
        # Compare:
        # If F_rescaled := avg_factor_sums * F,
        # Y_input ≈ C * F_rescaled
        #       Y ≈ C * F (Here, Y and F have normalized fibers)
        F_lateral_slices = eachslice(F, dims=2)
        F_lateral_slices .*= slice_sums
    end

    return C, F, rel_errors, norm_grad, dist_Ncone
end
#=
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

function updateC!(C, F, Y)
    @einsum FF[s,r] := F[s,j,k]*F[r,j,k]
    @einsum GG[i,r] := Y[i,j,k]*F[r,j,k]
    L = norm(FF)
    grad = C*FF .- GG
    C .-= grad ./ L # gradient step
    C .= ReLU.(C) # project
end

function updateF!(C, F, Y)
    CC = C'C
    L = norm(CC)
    grad = CC*F .- C'*Y
    F .-= grad ./ L # gradient step
    F .= ReLU.(F) # project
    #F_fibres = eachslice(F, dims=(1,2))
    #F_fibres .= projsplx.(F_fibres) # Simplex projection for each fibre in stead of ReLU
end

function calc_gradient(C, F, Y)
    @einsum FF[s,r] := F[s,j,k]*F[r,j,k]
    @einsum GG[i,r] := Y[i,j,k]*F[r,j,k]
    CC = C'C
    grad_C = C*FF .- GG
    grad_F = CC*F .- C'*Y
    return grad_C, grad_F
end
=#
# Could compute the gradients this way to reuse CF-Y,
# but the first way is still faster!
#=
CFY = C*F .- Y
@einsum grad_C[i,r] := CFY[i,j,k]*F[r,j,k]
@einsum grad_F[r,j,k] := C[i,r]*CFY[i,j,k]
return grad_C, grad_F
=#

"""Rescales C and F so each factor (horizontal slices) of F has similar magnitude."""
function rescaleCF2!(C, F)
    fiber_sums = sum.(eachslice(F, dims=1))
    #avg_factor_sums = mean.(eachrow(fiber_sums))

    F_horizontal_slices = eachslice(F, dims=1)
    F_horizontal_slices ./= fiber_sums

    C_rows = eachcol(C)
    C_rows .*= fiber_sums
end

function rescaleY2(Y)
    slice_sums = sum.(eachslice(Y, dims=1))
    #avg_fiber_sums = mean.(eachcol(fiber_sums))
    Yscaled = copy(Y)
    #Y_lateral_slices = eachslice(Yscaled, dims=2)
    Y ./= slice_sums
    return Yscaled, slice_sums
end
