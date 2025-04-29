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

    get!(kwargs, :constraints, nothing)

    original_constraints = kwargs[:constraints]

    constraints, kwargs = scale_constraints(Yₛ, coarsest_scale; continuous_dims, kwargs...)

    @info "Factorizing at scale $coarsest_scale..."
    decomposition, stats, _ = _factorize(Yₛ; default_kwargs(Yₛ; kwargs...)...) # calling the internal factorize so we don't get stuck in a loop calling factorize() <-> multiscale_factorize()

    # Factorize Y at progressively finer scales
    for scale in finer_scales
        # Use an interpolated version of the coarse factorization
        # as the initialization.
        # TODO generalize interpolate to not just doubling number of entries
        decomposition = interpolate(decomposition, 2; dims=continuous_dims, kwargs...)
        kwargs[:decomposition] = decomposition

        Yₛ = coarsen(Y, scale; dims=continuous_dims, kwargs...)

        # Reset constraints so they can be properly scaled
        kwargs[:constraints] = original_constraints
        constraints, kwargs = scale_constraints(Yₛ, scale; continuous_dims, kwargs...)

        @info "Factorizing at scale $scale..."
        decomposition, stats, _ = _factorize(Yₛ; default_kwargs(Yₛ; kwargs...)...) # calling the internal factorize so we don't get stuck in a loop calling factorize() <-> multiscale_factorize()
    end
    return decomposition, stats, kwargs
end

const IMPLEMENTED_DECOMPOSITION_CONSTRAINT_SCALING = [
    Tucker,
    Tucker1,
    CPDecomposition,
]

"""
    scale_constraints(Y, scale; kwargs...)

Scales any constraints that need to be modified to use at a coarser scale.
"""
function scale_constraints(Y, scale; continuous_dims, kwargs...)

    kwargs = Dict{Symbol,Any}(kwargs)

    kwargs[:continuous_dims] = continuous_dims
    S = log2(scale) + 1 # TODO Don't assume the scale is a power of 2
    # we want to map scales -> S where 1 -> 1, 2 -> 2, 4 -> 3, 8 -> 4, 16 -> 5 etc.

    decomposition, constraints = expand_decomposition_constraints(Y, kwargs)

    if !isnothing(constraints)
        if any(x-> typeof(decomposition) <: x, IMPLEMENTED_DECOMPOSITION_CONSTRAINT_SCALING)
            constraints = BlockedUpdate([scale_decomposition_constraint(continuous_dims, constraint, S, decomposition) for constraint in constraints])
        else
            @warn "Not sure how to appropriately scale constraints for a decomposition of type $decomposition. Leaving constraints alone."
        end
    end
    kwargs[:constraints] = constraints

    return kwargs[:constraints], kwargs
end

"""Use the same initialization as factorize() to get the expanded set of constraints"""
function expand_decomposition_constraints(Y, kwargs)
    kwargs_copy = deepcopy(kwargs) # Don't mess up anything since the following functions mutate kwargs
    kwargs_copy = default_kwargs(Y; kwargs_copy...) # TODO Is there some way to clean this up?
    decomposition, kwargs_copy = initialize_decomposition(Y; kwargs_copy...)
    expanded_constraints = parse_constraints(kwargs_copy[:constraints], decomposition; kwargs_copy...)
    return decomposition, expanded_constraints
end

# Continue to recurse
# function scale_constraint(; continuous_dims, constraint::BlockedUpdate, scale, decomposition)
#     return BlockedUpdate([scale_constraint(; continuous_dims, c, scale, decomposition) for c in constraint])
# end


# TODO extract info about internal vs external dimensions for AbstractDecompositions
# This uses the same pattern as interpolate(), so maybe there is a way to combine that info

"""
Idea is that external dimensions (I₁, I₂, ...) that are continuous dimensions need to be
scaled, but internal dimensions (R₁, R₂, ...) or non-continuous dimensions don't.
"""
function scale_decomposition_constraint(continuous_dims, constraint, scale, decomposition::CPDecomposition)
    if typeof(constraint) <: BlockedUpdate # cannot easily make this a separate method because we need this for any type of decomposition. TODO can this be cleaned up?
        return BlockedUpdate([scale_decomposition_constraint(continuous_dims, c, scale, decomposition) for c in constraint])
    end

    n = constraint.n

    # Find what dimensions the constraints apply over
    whats_constrained = try
            constraint.whats_normalized
        catch
            identityslice
        end
    slicemap = whats_constrained(factor(decomposition, n)).slicemap

    if n in 1:ndims(decomposition) # Constraint on a matrix
        if n in continuous_dims & slicemap[n] == Colon() # constraint applies to a continuous dimension & is sliced over
            n_continuous_dims = 1
            return scale_constraint(constraint, scale, n_continuous_dims)
        else # No need to scale the constraint
            return constraint
        end
    else
        error("Something went wrong and there is a constraint on factor $n which does not exist for CPDecomposition types.")
    end
end

# TODO use constraint.whats_normalized(decomposition).slicemap to get a tuple ex. (Colon(), 1, Colon()) to figure out which dimensions are being sliced over.
# Count how many of the non sliced (number of colons) dimensions are continuous to get n_continuous_dims

function scale_decomposition_constraint(continuous_dims, constraint, scale, decomposition::Tucker1)
    if typeof(constraint) <: BlockedUpdate # cannot easily make this a separate method because we need this for any type of decomposition. TODO can this be cleaned up?
        return BlockedUpdate([scale_decomposition_constraint(continuous_dims, c, scale, decomposition) for c in constraint])
    end

    # typeof(constraint) <: ConstraintUpdate || error("Expected a ConstraintUpdate type but got a constraint with type $(typeof(constraint))")

    n = constraint.n

    # Find what dimensions the constraints apply over
    whats_constrained = try
            constraint.whats_normalized
        catch
            identityslice
        end
    slicemap = whats_constrained(factor(decomposition, n)).slicemap

    if n == 0 # Constraint on the core
        sliced_dims = findall(slicemap[2:end] .== Colon()) .+ 1 # since we've removed the first entry slicemap[1], we need to shift the dimension labels back to what they are in slicemap
        # e.x., if a tensor of size (10, 33, 33) has continuous dimensions 2 and 3, and we slice over the 2nd and 3rd dimension
        # we get the slice map (1, Colon(), Colon()). After removing the first entry, we see slicemap[2:end] == (Colon(), Colon()), so findall(slicemap[2:end] .== Colon()) gives us [1, 2], not [2, 3].
        # note this is safe to empty lists since adding .+ 1 will keep it empty
        n_continuous_dims = count(d -> d in continuous_dims, sliced_dims) # how many continuous dimensions the constraint applies over
        if n_continuous_dims ≥ 1
            return scale_constraint(constraint, scale, n_continuous_dims)
        else
            return constraint
        end

    elseif n == 1 # Constraint on the matrix
        if (1 in continuous_dims) && (slicemap[1] == Colon()) # constraints apply over the first dimension
            n_continuous_dims = 1
            return scale_constraint(constraint, scale, n_continuous_dims)
        else # No need to scale the constraint
            return constraint
        end
    else
        error("Something went wrong and there is a constraint on factor $n which does not exist for Tucker1 types.")
    end
end

function scale_decomposition_constraint(continuous_dims, constraint, scale, decomposition::Tucker)
    if typeof(constraint) <: BlockedUpdate # cannot easily make this a separate method because we need this for any type of decomposition. TODO can this be cleaned up?
        return BlockedUpdate([scale_decomposition_constraint(continuous_dims, c, scale, decomposition) for c in constraint])
    end

    n = constraint.n

    # Find what dimensions the constraints apply over
    whats_constrained = try
            constraint.whats_normalized
        catch
            identityslice
        end
    slicemap = whats_constrained(factor(decomposition, n)).slicemap

    if n == 0 # Constraint on the core does not get scaled
        return constraint

    elseif n in 1:ndims(decomposition) # Constraint on a matrix
        if n in continuous_dims & slicemap[n] == Colon() # constraint applies to a continuous dimension & is sliced over
            n_continuous_dims = 1
            return scale_constraint(constraint, scale, n_continuous_dims)
        else # No need to scale the constraint
            return constraint
        end
    else
        error("Something went wrong and there is a constraint on factor $n which does not exist for Tucker types.")
    end
end

const SCALEABLE_CONSTRAINTS = [
    LinearConstraint,
    ScaledNormalization,
    ProjectedNormalization,
]

const FIXED_CONSTRAINTS = [
    Entrywise,
]

# Unpacks the constraint update and scales the internal AbstractConstraint
function scale_constraint(constraint::ConstraintUpdate, scale, n_continuous_dims)
    n = constraint.n

    # Convert the ConstraintUpdate to an AbstractConstraint
    constraint = getconstraint(constraint)::AbstractConstraint

    # Scale the constraint
    constraint = scale_constraint(constraint, scale, n_continuous_dims)

    # Convert back to a ConstraintUpdate
    constraint = ConstraintUpdate(n, constraint)
    return constraint
end
# Unpacks the constraint update and scales the internal AbstractConstraint
function scale_constraint(constraint::Rescale, scale, n_continuous_dims)
    n = constraint.n
    whats_rescaled = constraint.whats_rescaled

    # Convert the ConstraintUpdate to an AbstractConstraint
    constraint = getconstraint(constraint)::AbstractConstraint

    # Scale the constraint
    constraint = scale_constraint(constraint, scale, n_continuous_dims)

    # Convert back to a ConstraintUpdate
    constraint = Rescale(n, constraint, whats_rescaled)
    return constraint
end
# SafeNNProjection also needs a special translator
function scale_constraint(constraint::SafeNNProjection, scale, n_continuous_dims)
    n = constraint.n
    backup = constraint.backup

    # Scale the constraint
    constraint = scale_constraint(backup, scale, n_continuous_dims)

    # Convert back to a ConstraintUpdate
    constraint = SafeNNProjection(n, constraint)
    return constraint
end

"""
    scale_constraint(constraint::AbstractConstraint, scale, n_continuous_dims)

Returns a scaled version of the constraint based off the number of relevant continuous dimensions the constraint acts on.
"""
function scale_constraint(constraint::AbstractConstraint, scale, n_continuous_dims)
    @warn "Unsure how to scale constraints of type $(typeof(constraint)). Leaving the constraint $constraint alone."
    return constraint
end

# Constraints that are fixed like entrywise constraints do not need to be scaled
function scale_constraint(constraint::Union{FIXED_CONSTRAINTS...}, scale, n_continuous_dims)
    return constraint
end

# TODO handle LinearConstraint{<:AbstractArray} or LinearConstraint{Function}
function scale_constraint(constraint::LinearConstraint{<:AbstractMatrix}, scale, n_continuous_dims)
    A = constraint.linear_operator
    b = constraint.bias
    return LinearConstraint(A[:, begin:scale:end], b ./ scale^n_continuous_dims)
end

function scale_constraint(constraint::ScaledNormalization{<:Union{Real,AbstractArray{<:Real}}}, scale, n_continuous_dims)
    norm = constraint.norm
    F = constraint.whats_normalized
    S = constraint.scale
    return ScaledNormalization(norm, F, S ./ scale^n_continuous_dims)
end

function scale_constraint(constraint::ScaledNormalization{<:Function}, scale, n_continuous_dims)
    norm = constraint.norm
    F = constraint.whats_normalized
    S = constraint.scale
    return ScaledNormalization(norm, F, (x -> x ./ scale^n_continuous_dims) ∘ S)
end

function scale_constraint(constraint::ProjectedNormalization, scale, n_continuous_dims)
    @warn "Scaling ProjectedNormalization constraints is not implemented (YET!) Leaving the constraint $constraint alone."
    return constraint
end

function scale_constraint(constraint::ComposedConstraint, scale, n_continuous_dims)
    return scale_constraint(constraint.outer, scale, n_continuous_dims) ∘ scale_constraint(constraint.inner, scale, n_continuous_dims)
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
