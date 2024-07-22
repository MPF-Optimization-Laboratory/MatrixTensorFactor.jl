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
#superones(T::DataType, size::Integer, ndims::Integer=2) = SuperDiagonal(ones(T, size), ndims) # TODO make interface for a super diagonal of ones
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

function Base.size(S::SuperDiagonal{T,N}) where {T, N}
    n = length(diag(S))
    # Why Val(N) and not just N?
    # See https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-value-type
    return ntuple(i -> n, Val(N)) #Tuple(n for _ in 1:ndims(S))
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

getnotindex(A, i::Int) = A[eachindex(A) .!= i]
getnotindex(A, I) = A[eachindex(A) .∉ (I,)]

"""
    swapdims(A::AbstractArray, a::Integer, b::Integer=1)

Swap dimensions `a` and `b`.
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

"""Makes a Tuple of length n filled with `false`."""
false_tuple(n::Integer) = Tuple(fill(false, n))

"""
    projsplx(y::AbstractVector{<:Real})

Projects (in Euclidian distance) the vector y into the simplex.

[1] Yunmei Chen and Xiaojing Ye, "Projection Onto A Simplex", 2011
"""
function projsplx(y)
    n = length(y)

    if n==1 # quick exit for trivial length-1 "vectors" (i.e. scalars)
        return [one(eltype(y))]
    end

    y_sorted = sort(y[:]) # Vectorize/extract input and sort all entries
    i = n - 1
    t = 0 # need to ensure t has scope outside the while loop
    while true
        t = (sum(@view y_sorted[i+1:end]) - 1) / (n-i)
        if t >= y_sorted[i]
            break
        else
            i -= 1
        end

        if i >= 1
            continue
        else # i == 0
            t = (sum(y_sorted) - 1) / n
            break
        end
    end
    return ReLU.(y .- t)
end

"""max(0,x)"""
ReLU(x) = max(0,x)

"""
    identityslice(x::AbstractArray{T, N})

Useful for returning an iterable with a single iterate x
"""
function identityslice(A::AbstractArray{T, N}) where {T, N}
    # Why the Val(N) and not just N?
    # See https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-value-type
    ax = ntuple(dim -> Base.OneTo(1), Val(N))
    slicemap = ntuple(dim -> (:), Val(N))
    return Slices(A, slicemap, ax)
end

"""
    abs_randn(x...)

Folded normal or more specificly the half-normal initialization.
"""
abs_randn(x...) = abs.(randn(x...))

"""
    isnonnegative(X::AbstractArray{<:Real})
    isnonnegative(x::Real)

Checks if all entries of X are bigger or equal to zero.
"""
isnonnegative(X::AbstractArray{<:Real}) = all(isnonnegative, X)
isnonnegative(x::Real) = (x >= 0)

"""
    norm2(x)

L2 norm squared, the sum of squares of the entries of x.
"""
norm2(x) = mapreduce(abs2, +, x)

"""
    interlace(u, v)

Takes two iterables, u and v, and alternates elements from u and v into a vector.
If u and v are not the same length, extra elements are put on the end of the vector.
"""
function interlace(u, v)
    m = length(u)
    n = length(v)
    if m == n
        return vcat(collect.(zip(u,v))...)
    elseif m > n
        return vcat(interlace(u[begin:n], v), u[n+1:end])
    elseif m < n
        return vcat(interlace(u, v[begin:m]), v[m+1:end])
    end
end

"""
    geomean(v)
    geomean(v...)

Geometric mean of a collection of numbers: `prod(v)^(1/length(v))``.
"""
geomean(v) = prod(v)^(1/length(v))
geomean(v...) = geomean(v)

"""
    multifoldl(ops, args)

Like foldl, but with a different folding operation between each argument.
"""
function multifoldl(ops, args)
    @assert (length(ops) + 1) == length(args)
    x = args[begin]
    for (op, arg) in zip(ops, args[begin+1:end]) # TODO want @view args[begin+1:end] when possible
        x = op(x, arg)
    end
    return x
end
