# Functionality for performing factorize at multiple scales
# This is suitable for tensors that are discretizations of continuous data
# TODO Finish file

"""
    coarsen(Y::AbstractArray, scale::Integer; dims=1:ndims(Y))

Coarsens or downsamples `Y` by `scale`. Only keeps every `scale` entries along the dimensions specified.

Example
=======

Y = randn(12, 12, 12)

coarsen(Y, 2) == Y[begin:2:end, begin:2:end, begin:2:end]

coarsen(Y, 4; dims=(1, 3)) == Y[begin:4:end, :, begin:4:end]

coarsen(Y, 3; dims=2) == Y[:, begin:3:end, :]
"""
coarsen(Y::AbstractArray, scale::Integer; dims=1:ndims(Y), kwargs...) =
    Y[(d in dims ? axis[begin:scale:end] : axis for (d, axis) in enumerate(axes(Y)))...]

# Using axis[begin:scale:end] rather than 1:scale:size(Y, d) to allow for more flexible indexing

# function coarsen(Y::Array, scale::Integer; dims=1:ndims(Y), kwargs...)
#     sizes = size(Y)

#     # Assumes 1:I based indexing
#     # Y[(d in dims ? (begin:scale:end) : (:) for d in 1:N)...] does not work here
#     # since Julia treats "end" as length(Y), not the length of the corresponding dimension
#     return Y[(d in dims ? (1:scale:s) : (:) for (d, s) in enumerate(sizes))...]
# end

# # More expensive method since we need to convert Y to a string and then parse it back
# # But this should work on any AbstractArray
# function coarsen(Y::AbstractArray, scale::Integer; dims=1:ndims(Y), kwargs...)
#     N = ndims(Y)

#     slice = join((d in dims ? "begin:$scale:end" : ":" for d in 1:N), ",")

#     Y_coarsened = eval(Meta.parse("$Y[$(slice)]"))

#     return Y_coarsened
# end
