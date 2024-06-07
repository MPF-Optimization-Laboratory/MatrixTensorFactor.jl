"""
Different tensor products. Note these do not use Einsum/Tullio or any entrywise notation
since these are defined generaly for arbitrarily ordered tensors/arrays.
"""

"""
    khatrirao(A::AbstractMatrix, B::AbstractMatrix)
    A ⊙ B

Khatri-Rao product of two matricies. A ⊙ B can be typed with `\\odot`.
"""
khatrirao(A::AbstractMatrix, B::AbstractMatrix) = hcat(kron.(eachcol(A), eachcol(B))...)
const ⊙ = khatrirao
