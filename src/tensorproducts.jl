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

nmp = nmode_product # Short-hand alias

"""
    swapdims(A::AbstractArray, a::Integer, b::Integer=1)

Swap dimentions `a` and `b`.
"""
function swapdims(A::AbstractArray, a::Integer, b::Integer=1)
    dims = collect(1:ndims(A)) # TODO construct the permutation even more efficiently
    dims[a] = b; dims[b] = a
    permutedims(A, dims)
end

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
    Cmat = A * Bmat
    C = reshape(Cmat, size(A)[1], sizeB[2:end]...)
    return C
end

"""
    khatrirao(A::AbstractMatrix, B::AbstractMatrix)
    A ⊙ B

Khatri-Rao product of two matricies. A ⊙ B can be typed with `\\odot`.
"""
khatrirao(A::AbstractMatrix, B::AbstractMatrix) = hcat(kron.(eachcol(A), eachcol(B))...)
const ⊙ = khatrirao
