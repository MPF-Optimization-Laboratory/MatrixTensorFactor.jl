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
    interpolate(Y, scale; dims=1:ndims(Y), degree=0, kwargs...)

Interpolates Y to a larger array with repeated values.

Keywords
========
`scale`. How much to scale up the size of `Y`.
A dimension with size `k` will be scaled to `scale*k - (scale - 1) = scale*(k-1) + 1`

`dims`:`1:ndims(Y)`. Which dimensions to interpolate.

`degree`:`0`. What degree of interpolation to use. `0` is constant interpolation, `1` is linear.

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
function interpolate(Y::AbstractArray, scale; dims=1:ndims(Y), degree=0, kwargs...)
    # Quick exit if no interpolation is needed
    if scale == 1 || isempty(dims)
        return Y
    end

    Y = repeat(Y; inner=(d in dims ? scale : 1 for d in 1:ndims(Y)))

    # Chop the last slice of repeated dimensions since we only interpolate between
    # the values
    chop = (d in dims ? axis[begin:end-scale+1] : axis for (d, axis) in enumerate(axes(Y)))
    Y = Y[chop...]

    if degree == 0
        return Y
    elseif degree == 1 && scale == 2 # TODO generalize linear_smooth to other scales
        return linear_smooth(Y; dims, kwargs...)
    else
        error("interpolation of degree=$degree with scale=$scale not supported (YET!)")
    end
end

function interpolate(A::AbstractDecomposition, scale; kwargs...)
    error("Not sure how to interpolate $(typeof(Y)). (YET!)") # TODO make a general way of interpolating any decomposition
end

function interpolate(CPD::CPDecomposition, scale; dims=1:ndims(CPD), kwargs...)
    interpolated_matrix_factors = (d in dims ? interpolate(A, scale; dims=1, kwargs...) : A for (d, A) in enumerate(matrix_factors(CPD))) # TODO make an each_matrix_factor_index iterator?
    return CPDecomposition(Tuple(interpolated_matrix_factors))
end

function interpolate(T::Tucker1, scale; dims=1:ndims(T), kwargs...)
    core_dims = setdiff(dims, 1) # Want all dimensions except possibly the first
    interpolated_core = interpolate(core(T), scale; dims=core_dims, kwargs...)

    matrix = matrix_factor(T, 1)

    interpolated_matrix = 1 in dims ? interpolate(matrix, scale; dims=1, kwargs) : matrix
    return Tucker1((interpolated_core, interpolated_matrix))
end

function interpolate(T::Tucker, scale; dims=1:ndims(T), kwargs...)
    interpolated_matrix_factors = (d in dims ? interpolate(A, scale; dims=1, kwargs...) : A for (d, A) in enumerate(matrix_factors(T))) # TODO make an each_matrix_factor_index iterator?
    # Core is not interpolated
    return Tucker(Tuple(core(T), interpolated_matrix_factors...))
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

"""
    multiscale_factorize(Y; continuous_dims=1:ndims(Y), rank=1, model=Tucker1, kwargs...)

Like [`factorize`](@ref) but uses progressively finer sub-grids of `Y` to speed up convergence. This is only effective when the dimensions given by `dims` come from discretizations of continuous data.

For example, if `Y` has 3 dimensions where `Y[i, j, k]` are samples from a continuous 2D function f_i(x_j, y_k) on a grid, use `multiscale_factorize(Y; continuous_dims=(2,3))` since second and third dimensions are continuous.
"""
function multiscale_factorize(Y; kwargs...)
    continuous_dims, kwargs = initialize_continuous_dims(Y; kwargs...)
    scales, kwargs = initialize_scales(Y; kwargs...)
    coarsest_scale, finer_scales... = scales

    # Factorize Y at the coarsest scale
    Yₛ = coarsen(Y, coarsest_scale; dims=continuous_dims, kwargs...)

    decomposition, stats, _ = factorize(Yₛ; kwargs...)

    # Factorize Y at progressively finer scales
    for scale in finer_scales
        # Use an interpolated version of the coarse factorization
        # as the initialization.
        # TODO generalize interpolate to not just doubling number of entries
        decomposition = interpolate(decomposition, 2; dims=continuous_dims, kwargs...)
        kwargs[:decomposition] = decomposition

        Yₛ = coarsen(Y, scale; dims=continuous_dims, kwargs...)

        decomposition, stats, _ = factorize(Yₛ; decomposition, kwargs...)
    end
    return decomposition, stats, kwargs
end


"""
    initialize_scales(Y, kwargs)

Initializes the plan for factorizing at progressively finer scales.

The list of scales should be ordered from largest (coarse) to smallest (fine) and end with 1.
"""
function initialize_scales(Y; continuous_dims, kwargs...)
    kwargs = Dict{Symbol,Any}(kwargs)
    get!(kwargs, :scales, nothing)

    # Quick exit if scales have already been defined
    # TODO add some checking to make sure scales is valid
    if !isnothing(kwargs[:scales])
        return kwargs[:scales], kwargs
    end

    continuous_dimension_sizes = (size(Y, d) for d in continuous_dims)

    all(ispower_of_2_plus_one, continuous_dimension_sizes) ||
        error("`multiscale_factorize` can only handle continuous dimensions that are one plus a power of 2 (as of yet!)")
        # TODO allow for more general multiscale plans

    I_min = minimum(continuous_dimension_sizes)
    I_min ≥ 3 || error("Smallest continuous dimension must be at least 3")

    S_min = Int(log2(I_min - 1))

    kwargs[:scales] = (2^(S_min - s) for s in 0:S_min)

    return kwargs[:scales], kwargs
end

ispower_of_2_plus_one(x) = isinteger(log2(x - 1))

"""
    initialize_continuous_dims(Y; kwargs...)

Lists dimensions of Y that represent a discretization of a continuous function.

Defaults to all of them: `continuous_dims = 1:ndims(Y)`.
"""
function initialize_continuous_dims(Y; kwargs...)
    # This is the first initializing function so we need to check if there are any kwargs at all
    kwargs = isempty(kwargs) ? Dict{Symbol,Any}() : Dict{Symbol,Any}(kwargs)

    # TODO add smarter default that checks which dimensions look continuous
    get!(kwargs, :continuous_dims, 1:ndims(Y))
    return kwargs[:continuous_dims], kwargs
end
