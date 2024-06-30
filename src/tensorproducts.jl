"""
Different tensor products. Note these do not use Einsum/Tullio or any entrywise notation
since these are defined generaly for arbitrarily ordered tensors/arrays.
"""

"""
    nmode_product(A::AbstractArray, B::AbstractMatrix, n::Integer)

This is defined for arbitrary n, which cannot be done using Einsum/Tullio since they can
only handle fixed ordered-tensors.
"""
function nmode_product(A::AbstractArray, B::AbstractMatrix, n::Integer)
    Aperm = swapdims(A, n)
    Cperm = Aperm ×₁ B # convert the problem to the mode-1 product
    return swapdims(Cperm, n) # swap back
    #Cmat = B * mat(A, n)
    #sizeA = size(A)
    #sizeC = (sizeA[begin:n-1]..., size(B)[2], sizeA[n+1:end]...)
    #return imat(Cmat, n, sizeC)
end

function nmode_product(A::AbstractArray, b::AbstractVector, n::Integer)
    Aperm = swapdims(A, n)
    Cperm = Aperm ×₁ b # convert the problem to the mode-1 product
    return Cperm # no need to swap since the first dimention is dropped
end

const nmp = nmode_product # Short-hand alias

"""1-mode product between a tensor and a matrix"""
×₁(A::AbstractArray, B::AbstractMatrix) = mtt(B, A)
×₁(A::AbstractArray, b::AbstractVector) = dropdims(mtt(b', A); dims=1)

"""
Matrix times Tensor

Looks like C[i1, i2, ..., iN] = sum_r A[i1, r] * B[r, i2, ..., iN] entry-wise.
"""
function mtt(A::AbstractMatrix, B::AbstractArray)
    sizeB = size(B)
    Bmat = reshape(B, sizeB[1], :)

    #Cmat = A * Bmat
    #C = reshape(Cmat, size(A)[1], sizeB[2:end]...)

    # Slightly faster implimentation
    C = zeros(size(A)[1], sizeB[2:end]...)
    Cmat = reshape(C, size(A)[1], prod(sizeB[2:end]))
    mul!(Cmat, A, Bmat)

    return C
end

# TODO boost performance of slicewise_dot by:
# swapping the dimentions
"""
    slicewise_dot(A::AbstractArray, B::AbstractArray; dims=1, dimsA=dims, dimsB=dims)

Constracts all but the dimentions `dimsA` and `dimsB` of A and B by performing a dot product over each `dim=dimsX` slice.

Generalizes `@einsum C[s,r] := A[s,j,k]*B[r,j,k]` to arbitrary dimentions.

For example, if A and B are both matricies, slicewise_dot(A, B) == A*B'
"""
function slicewise_dot(A::AbstractArray, B::AbstractArray; dims=1, dimsA=dims, dimsB=dims)
    C = zeros(size(A)[dimsA], size(B)[dimsB]) # Array{promote_type(T, U), 2}(undef, size(A)[1], size(B)[1]) doesn't seem to be faster

    if A === B && (dimsA == dimsB)# use the faster routine if they are the same array
        return _slicewise_self_dot!(C, A; dims=dimsA)
    end

    for (i, A_slice) in enumerate(eachslice(A, dims=dimsA))
        for (j, B_slice) in enumerate(eachslice(B, dims=dimsB))
            C[i, j] = A_slice ⋅ B_slice
        end
    end
    return C
end

function _slicewise_self_dot!(C, A; dims=1)
    enumerated_A_slices = enumerate(eachslice(A; dims))
    for (i, Ai_slice) in enumerated_A_slices
        for (j, Aj_slice) in enumerated_A_slices
            if i > j
                continue
            else
                C[i, j] = Ai_slice ⋅ Aj_slice
            end
        end
    end
    return Symmetric(C)
end

"""
    tucker_contractions(N)

Contractions used in a full Tucker decomposition of order N. This is a tuple of the n-mode
products from 1 to N in order.
"""
tucker_contractions(N) = Tuple((G, A) -> nmp(G, A, n) for n in 1:N)

"""
    tuckerproduct(G, (A, B, ...))
    tuckerproduct(G, A, B, ...)

Multiplies the inputs by treating the first argument as the core and the rest of the
arguments as matricies in a Tucker decomposition.

Example
-------
tuckerproduct(G, A, B, C) == G ×₁ A ×₂ B ×₃ C
tuckerproduct(G, A, C; exclude=2) == G ×₁ A ×₃ C
"""
function tuckerproduct(core, matricies; exclude=nothing)
    N = ndims(core)
    if isnothing(exclude)
        N == length(matricies) ||
            throw(ArgumentError("expected $N number of matricies, got $(length(matricies))"))
        return multifoldl(tucker_contractions(N), matricies)
    else
        N == length(matricies) - length(exclude) ||
        throw(ArgumentError("expected $N number of matricies, got $(length(matricies))"))
        return multifoldl(getnotindex(tucker_contractions(N), exclude), matricies)
    end
end
tuckerproduct(core, matricies...; kwargs...) = tuckerproduct(core, matricies; kwargs...)

"""
    cpproduct((A, B, C, ...))
    cpproduct(A, B, C, ...)

Multiplies the inputs by treating them as matricies in a CP decomposition.

Example
-------
cpproduct(A, B, C) == @einsum T[i, j, k] := A[i, r] * B[j, r] * C[k, r]
"""
cpproduct(matricies) = mapreduce(vector_outer, +, zip((eachcol.(matricies))...))
cpproduct(matricies...) = cpproduct(matricies)

"""
    khatrirao(A::AbstractMatrix, B::AbstractMatrix)
    A ⊙ B

Khatri-Rao product of two matricies. A ⊙ B can be typed with `\\odot`.
"""
khatrirao(A::AbstractMatrix, B::AbstractMatrix) = hcat(kron.(eachcol(A), eachcol(B))...)
const ⊙ = khatrirao
