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

# Using axis[begin:scale:end] rather than 1:scale:size(Y, d) for more flexible indexing

"""
    interpolate(Y, scale; dims=1:ndims(Y), kwargs...)

Interpolates Y to a larger array with repeated values.

Like the opposite of [`coarsen`](@ref).

Example
=======

julia> Y = collect(reshape(1:6, 2, 3))
2×3 Matrix{Int64}:
 1  3  5
 2  4  6

julia> interpolate(Y, 2)
3×5 Matrix{Int64}:
 1  1  3  3  5
 1  1  3  3  5
 2  2  4  4  6

julia> interpolate(Y, 3; dims=2)
2×7 Matrix{Int64}:
 1  1  1  3  3  3  5
 2  2  2  4  4  4  6

julia> interpolate(Y, 1) == Y
true
"""
function interpolate(Y, scale; dims=1:ndims(Y), kwargs...)
    Y = repeat(Y; inner=(d in dims ? scale : 1 for d in 1:ndims(Y)))

    # Chop the last slice of repeated dimensions since we only interpolate between
    # the values
    return Y[(d in dims ? axis[begin:end-scale+1] : axis for (d, axis) in enumerate(axes(Y)))...]
end

# TODO add more intelligent interpolation e.g. linear
# TODO add tests for interpolate
# TODO do we want linear_smooth(linear_smooth(Z)) == linear_smooth(Z) ?
# right now this linear smooth only works on arrays that came from interpolate(Y, 2)

function linear_smooth(Y; dims=1:ndims(Y), kwargs...)
    return _linear_smooth!(1.0 * Y, dims)
    # makes a copy of Y and ensures the type can hold float like elements
end

function _linear_smooth!(Y, dims)
    all_dims = 1:ndims(Y)
    for d in dims
        axis = axes(Y, d)
        Y1 = @view Y[(i==d ? axis[begin+1:end-1] : (:) for i in all_dims)...]
        Y2 = @view Y[(i==d ? axis[begin+2:end] : (:) for i in all_dims)...]

        @. Y1 = 0.5 * (Y1 + Y2)
    end
    return Y
end


# A = Array{eltype(Y)}(undef,(size(Y).* scale .- 1)...)
