"""
Utility functions used throughout the package that don't fit anywhere else
"""

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
