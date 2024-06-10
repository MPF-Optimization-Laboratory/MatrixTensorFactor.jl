"""
Utility functions used throughout the package that don't fit anywhere else
"""

"""
    SuperDiagonal{T, N} <: AbstractArray{T,N}

Array of order N that is zero everywhere except possibly along the super diagonal.
"""
struct SuperDiagonal{T, N, V<:AbstractVector{T}} <: AbstractArray{T,N}
    diag::V

    function SuperDiagonal{T, N, V}(diag) where {T, N, V<:AbstractVector{T}}
        Base.require_one_based_indexing(diag)
        new{T, N, V}(diag)
    end
end

# SuperDiagonal Interface
SuperDiagonal(v::AbstractVector, ndims::Integer=2) = SuperDiagonal{eltype(v), ndims, typeof(v)}(v)
SuperDiagonal{T}(v::AbstractVector, ndims::Integer=2) where {T} = SuperDiagonal(convert(AbstractVector{T}, v)::AbstractVector{T}, ndims)
LinearAlgebra.diag(S::SuperDiagonal) = S.diag
function array(S::SuperDiagonal)
    A = zeros(eltype(S), size(S))
    A[_diagonal_indexes(S)] .= diag(S)
    return A
end
_diagonal_indexes(S::SuperDiagonal) = CartesianIndex.(fill(1:length(diag(S)), ndims(S))...)

# AbstractArray interface
#Base.ndims(_::SuperDiagonal{T,N}) where {T,N} = N

function Base.size(S::SuperDiagonal)
    n = length(diag(S))
    return Tuple(n for _ in 1:ndims(S))
end

Base.getindex(S::SuperDiagonal, i::Int) = getindex(array(S), i) # TODO work out this efficiently
function Base.getindex(S::SuperDiagonal, I::Vararg{Int})
    # Check if index is valid
    min, max = extrema(I)
    n = length(diag(S))

    if min < 1 || n < max
        throw(BoundsError("attempt to access $(size(S)) SuperDiagonal at index $I"))
    elseif allequal(I)
        return getindex(diag(S), I[begin])
    else
        return zero(eltype(S))
    end
end

#######################################################

"""
    swapdims(A::AbstractArray, a::Integer, b::Integer=1)

Swap dimentions `a` and `b`.
"""
function swapdims(A::AbstractArray, a::Integer, b::Integer=1)
    dims = collect(1:ndims(A)) # TODO construct the permutation even more efficiently
    dims[a] = b; dims[b] = a
    permutedims(A, dims)
end

"""
mat(A::AbstractArray, n::Integer)

Matricize along the nth mode.
"""
function mat(A::AbstractArray, n::Integer)
    N = ndims(A)
    1 <= n && n <= N || throw(ArgumentError("n=$n is not a valid dimention to matricize"))
    dims = ((1:n-1)..., (n+1:N)...) # all axis, skipping n
    return cat(eachslice(A; dims)...; dims=2) # TODO Idealy return a view/SubArray
end

#fullouter(v...) = reshape(kron(reverse(vec.(v))...),tuple(vcat(collect.(size.(v))...)...))
