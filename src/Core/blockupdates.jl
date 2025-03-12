"""
Mid level code that combines constraints with block updates to be used on an AbstractDecomposition
"""


"""
Interface to make a step scheme is

struct MyStep <: AbstractStep
    ...
end

function (step::MyStep)(x::AbstractDecomposition; kwargs...)
    ...
    return step::Real
end

To use your scheme, construct an instance with any necessary parameters

mystep = MyStep(...)

and then you can call

step = mystep(D; kwargs...)

to compute the step size.
"""
abstract type AbstractStep <: Function end

struct LipschitzStep <: AbstractStep
    lipschitz::Function
end

function (step::LipschitzStep)(x; kwargs...)
    L = step.lipschitz(x)
    try
        return L^(-1)  # allow for Lipschitz to be a diagonal matrix
    catch
        @warn "Could not invert the Lipschitz constant to get a stepsize. Ignoring zero coordinates."

        return _safe_invert.(L)
    end
end

_safe_invert(x) = iszero(x) ? x : x^(-1)

function (step::LipschitzStep)(x::Tucker; kwargs...)
    L = step.lipschitz(x)
    if typeof(L) <: Tuple # Currently the only case is when we are updating the core of a Tucker factorization
                          # Using this condition as a way to tell if it is the core we are calculating the constant for
        return map(X -> X^(-1), L)
    else
        return L^(-1) # allow for Lipschitz to be a diagonal matrix
    end
end
#LipschitzStep(L::Real) = 1/L

# TODO have these be functions that act on decompositions more generally

function make_lipschitz(T::Tucker1, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    if n==0 # the core is the zeroth factor
        function lipschitz0(T::Tucker1; kwargs...)
            A = matrix_factor(T, 1)
            return opnorm(A'A)
        end
        return lipschitz0

    elseif n==1 # the matrix is the zeroth factor
        function lipschitz1(T::Tucker1; kwargs...)
            C = core(T)
            return opnorm(slicewise_dot(C, C))
        end
        return lipschitz1

    else
        error("No $(n)th factor in Tucker1")
    end
end

function make_lipschitz(T::Tucker, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n==0 # the core is the zeroth factor
        function lipschitz_core(T::AbstractTucker; kwargs...)
            #matrices = matrix_factors(T)
            #gram_matrices = map(A -> A'A, matrices)
            #return prod(opnorm.(gram_matrices))
            return prod(A -> opnorm(A'A), matrix_factors(T))
        end
        return lipschitz_core

    elseif n in 1:N # the matrix is the zeroth factor
        function lipschitz_matrix(T::AbstractTucker; kwargs...)
            matrices = matrix_factors(T)
            TExcludeAn = tuckerproduct(core(T), matrices; exclude=n)
            return opnorm(slicewise_dot(TExcludeAn, TExcludeAn; dims=n))
        end
        return lipschitz_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end

function make_lipschitz(T::CPDecomposition, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n in 1:N # the matrix is the zeroth factor
        function lipschitz_matrix(T::AbstractTucker; kwargs...)
            matrices = matrix_factors(T)
            TExcludeAn = tuckerproduct(core(T), matrices; exclude=n) # TODO optimize this to avoid making the super diagonal core
            return opnorm(slicewise_dot(TExcludeAn, TExcludeAn; dims=n))
        end
        return lipschitz_matrix

    else
        error("No $(n)th factor in CPDecomposition")
    end
end

function make_block_lipschitz(T::Tucker1, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    if n==0 # the core is the zeroth factor
        function lipschitz0(T::Tucker1; kwargs...)
            A = matrix_factor(T, 1)
            return Diagonal_col_norm(A'A) # Diagonal(norm2.(eachcol(A)))
        end
        return lipschitz0

    elseif n==1 # the matrix is the zeroth factor
        function lipschitz1(T::Tucker1; kwargs...)
            C = core(T)
            return Diagonal_col_norm(slicewise_dot(C, C)) #Diagonal(norm2.(eachslice(C; dims=1)))
        end
        return lipschitz1

    else
        error("No $(n)th factor in Tucker1")
    end
end

function make_block_lipschitz(T::CPDecomposition, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n in 1:N # the matrix is the zeroth factor
        function lipschitz_matrix(T::AbstractTucker; kwargs...)
            matrices = matrix_factors(T)
            TExcludeAn = tuckerproduct(core(T), matrices; exclude=n) # TODO optimize this to avoid making the super diagonal core
            return Diagonal_col_norm(slicewise_dot(TExcludeAn, TExcludeAn; dims=n))
        end
        return lipschitz_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end

function make_block_lipschitz(T::Tucker, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n==0 # the core is the zeroth factor
        function lipschitz_core(T::AbstractTucker; kwargs...)
            return map(A -> Diagonal_col_norm(A'A), matrix_factors(T)) # Return a tuple of diagonal matrices
        end
        return lipschitz_core

    elseif n in 1:N # the matrix is the zeroth factor
        function lipschitz_matrix(T::AbstractTucker; kwargs...)
            matrices = matrix_factors(T)
            TExcludeAn = tuckerproduct(core(T), matrices; exclude=n)
            return Diagonal_col_norm(slicewise_dot(TExcludeAn, TExcludeAn; dims=n))
        end
        return lipschitz_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end

struct ConstantStep <: AbstractStep
    stepsize::Real
end

(step::ConstantStep)(x; kwargs...) = step.stepsize

struct SPGStep <: AbstractStep
    min::Real
    max::Real
end

SPGStep(;min=1e-10, max=1e10) = SPGStep(min, max)

# Convert an input of the full decomposition, to a calculation on the factor
# Calculate the last gradient if a function was provided
(step::SPGStep)(x::T; n, x_last::T, grad_last::Function, kwargs...) where {T <: AbstractDecomposition} =
    step(factor(x,n); x_last=factor(x_last,n), grad_last=grad_last(x_last), kwargs...)

# option to override the set defaults from step
# TODO SPG has a linesearch/negative momentum update part to the fill iteration
# but in the best case, this linesearch just uses the value given by this step
# so I will skip implementing it for now, but may want to add that once
# I add a line search
function (step::SPGStep)(x; grad, x_last, grad_last, stepmin=step.min, stepmax=step.max, kwargs...)
    s = x - x_last
    y = grad - grad_last
    sy = (s ⋅ y)
    if sy <=0 #TODO check why (s ⋅ y) < 0 means we should take stepmax and not stepmin
        return stepmax
    else
        suggested_step = (s ⋅ s) / sy
        return clamp(suggested_step, stepmin, stepmax) # safeguards to ensure step is within reasonable bounds
    end
end


###########################


abstract type AbstractUpdate <: Function end

function Base.show(io::IO, x::AbstractUpdate)
    print(io, typeof(x))
    print(io, "(")
    data = Any[]
    for p in propertynames(x)
        if p in (:n, :proj, :scale) # Most verbose
            push!(data, getproperty(x, p))
        elseif p in (:step, ) # Only show the type of this property
            push!(data, typeof(getproperty(x, p)))
        else # Least verbose, just give the name of this property
            push!(data, p)
        end
    end
    join(io, data, ", ")
    print(io, ")")
end

struct GenericUpdate <: AbstractUpdate
    f::Function
end

(U::GenericUpdate)(x; kwargs...) = U.f(x; kwargs...)

function Base.getproperty(U::AbstractUpdate, sym::Symbol)
    if sym === :n
        try
            return getfield(U, :n)
        catch e
            @debug "Got $e. $U does not have an assigned factor `n` it updates."
            return nothing
        end
    else # fallback to other fields
        return getfield(U, sym)
    end
end

function checkfrozen(x, n)
    frozen = isfrozen(x, n)
    if frozen
        @debug "Factor $n is frozen, skipping its update."
    end
    return frozen
end

abstract type AbstractGradientDescent <: AbstractUpdate end

"""
Perform a Gradient decent step on the nth factor of an Abstract Decomposition x

The n is only to keep track of the factor that gets updated, and to check if a frozen
factor was requested to be updated.
"""
struct GradientDescent <: AbstractGradientDescent
    n::Integer
    gradient::Function
    step::AbstractStep
end

function (U::GradientDescent)(x; x_last, kwargs...)
    n = U.n
    if checkfrozen(x, n)
        return x
    end
    grad = U.gradient(x; kwargs...)
    # Note we pass a function for grad_last (lazy) so that we only compute it if needed for the step
    s = U.step(x; n, x_last, grad, grad_last=(x -> U.gradient(x; kwargs...)), kwargs...)
    a = factor(x, n)
    @. a -= s*grad
end

function make_gradient(D::AbstractDecomposition, n::Integer, Y::AbstractArray; objective::AbstractObjective, kwargs...)
    error("Gradient not implemented for ", typeof(D), " with ", typeof(objective), " objective")
end

# Using this pattern of inputs so that gradients for a generic decomposition could be calculated
# with auto diff by looking at the gradient of the function objective(D, Y) with respect to the nth factor in D
function make_gradient(T::Tucker1, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    if n==0 # the core is the zeroth factor
        function gradient0(X::Tucker1; kwargs...)
            (B, A) = factors(X)
            AA = A'A
            YA = Y×₁A'
            grad = B×₁AA - YA
            return grad
        end
        return gradient0
    elseif n==1 # the matrix is the first factor
        function gradient1(X::Tucker1; kwargs...)
            (B, A) = factors(X)
            BB = slicewise_dot(B, B)
            YB = slicewise_dot(Y, B)
            grad = A*BB - YB
            return grad
        end
        return gradient1
    else
        error("No $(n)th factor in Tucker1")
    end
end

function make_gradient(T::Tucker, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n==0 # the core is the zeroth factor
        function gradient_core(X::AbstractTucker; kwargs...)
            B = core(X)
            matrices = matrix_factors(X)
            gram_matrices = map(A -> A'A, matrices) # gram matrices AA = A'A,
                                                    # BB = B'B, ...
            grad = tuckerproduct(B, gram_matrices)
                 - tuckerproduct(Y, adjoint.(matrices))
            return grad
        end
        return gradient_core

    elseif n in 1:N # the matrix factors start at m=1
        function gradient_matrix(X::AbstractTucker; kwargs...)
            B = core(X)
            matrices = matrix_factors(X)
            Aₙ = factor(X, n)
            X̃ₙ = tuckerproduct(B, matrices; exclude=n)
            grad = Aₙ * slicewise_dot(X̃ₙ, X̃ₙ; dims=n)
                   - slicewise_dot(Y, X̃ₙ; dims=n)
            return grad
        end
        return gradient_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end

function make_gradient(T::CPDecomposition, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n in 1:N # the matrix factors start at m=1
        function gradient_matrix(X::AbstractTucker; kwargs...)
            B = core(X)
            matrices = matrix_factors(X)
            Aₙ = factor(X, n)
            X̃ₙ = tuckerproduct(B, matrices; exclude=n)
            grad = Aₙ * slicewise_dot(X̃ₙ, X̃ₙ; dims=n)
                   - slicewise_dot(Y, X̃ₙ; dims=n)
            return grad
        end
        return gradient_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end

"""
Perform a Block Gradient decent step on the nth factor of an Abstract Decomposition x

The n is only to keep track of the factor that gets updated, and to check if a frozen
factor was requested to be updated.

This type allows for more complicated step sizes such as individual steps for sub-blocks of
the nth factor.
"""
struct BlockGradientDescent <: AbstractGradientDescent
    n::Integer
    gradient::Function
    step::AbstractStep
    combine::Function # takes a step (number, matrix, or tensor) and combines it with a gradient
end

function (U::BlockGradientDescent)(x; x_last, kwargs...)
    n = U.n
    if checkfrozen(x, n)
        return x
    end
    grad = U.gradient(x; kwargs...)
    # Note we pass a function for grad_last (lazy) so that we only compute it if needed for the step
    s = U.step(x; n, x_last, grad, grad_last=(x -> U.gradient(x; kwargs...)), kwargs...)
    a = factor(x, n)
    a .-= U.combine(grad, s)
end

function make_blockGD_combines(T::Tucker1, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    if n==0 # the core is the zeroth factor
        combine0(grad, step) = grad ×₁ step # need to multiply grad (a tensor) by the Lipschitz matrix
        return combine0

    elseif n==1 # the matrix is the zeroth factor
        combine1(grad, step) = grad * step # need right matrix multiplication
        return combine1

    else
        error("No $(n)th factor in Tucker1")
    end
end

function make_blockGD_combines(T::Tucker, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n==0 # the core is the zeroth factor
        combine_core(grad, step) = tuckerproduct(grad, step) # need to multiply grad (a tensor) by each Lipschitz matrix
        return combine_core

    elseif n in 1:N # the matrix factors start at m=1
        combine_matrix(grad, step) = grad * step # need right matrix multiplication
        return combine_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end

function make_blockGD_combines(T::CPDecomposition, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n in 1:N # the matrix is the zeroth factor
        combine_matrix(grad, step) = grad * step # need right matrix multiplication
        return combine_matrix

    else
        error("No $(n)th factor in Tucker1")
    end
end

abstract type ConstraintUpdate <: AbstractUpdate end

"""
    ConstraintUpdate(n, constraint)

Converts an AbstractConstraint to a ConstraintUpdate on the factor n
"""
ConstraintUpdate(n, constraint::AbstractConstraint; kwargs...) = error("converting $(typeof(constraint)) to a ConstraintUpdate is not yet supported")
ConstraintUpdate(n, constraint::GenericConstraint; kwargs...) = GenericConstraintUpdate(n, constraint)
ConstraintUpdate(n, constraint::ProjectedNormalization; kwargs...) = Projection(n, constraint)
ConstraintUpdate(n, constraint::Entrywise; kwargs...) = Projection(n, constraint)

function ConstraintUpdate(n, constraint::ScaledNormalization; skip_rescale=false, whats_rescaled=missing, kwargs...)
    if skip_rescale
        ismissing(whats_rescaled) || isnothing(whats_rescaled) || @warn "skip_rescale=true but whats_rescaled=$whats_rescaled was given. Overriding to whats_rescaled=nothing"
        return Rescale(n, constraint, nothing)
    else
        return Rescale(n, constraint, whats_rescaled)
    end
end

function ConstraintUpdate(n, constraint::ComposedConstraint; kwargs...)
    if constraint.inner == nonnegative! && typeof(constraint.outer) <: ScaledNormalization
        norm = constraint.outer.norm
        return BlockedUpdate(
            SafeNNProjection(n, ProjectedNormalization(norm, makeNNprojection(norm); whats_normalized=constraint.outer.whats_normalized)),
            ConstraintUpdate(n, constraint.outer; kwargs...))
    else
        return BlockedUpdate(ConstraintUpdate(n, constraint.inner; kwargs...), ConstraintUpdate(n, constraint.outer; kwargs...)) # note we apply inner constraint first
    end
end

check(_::ConstraintUpdate, _::AbstractDecomposition) = error("checking $(typeof(constraint)) is not yet supported")

struct GenericConstraintUpdate <: ConstraintUpdate
    n::Integer
    constraint::GenericConstraint
end

check(U::GenericConstraintUpdate, D::AbstractDecomposition) = check(U.constraint, factor(D, U.n))

function (U::GenericConstraintUpdate)(x::T; kwargs...) where T
    n = U.n
    if checkfrozen(x, n)
        return x
    end
    A = factor(x, n)
    U.constraint(A)
    check(U, A) || error("Something went wrong with GenericConstraintUpdate: $GenericConstraintUpdate")
end

"""Perform a projected gradient update on the nth factor of an Abstract Decomposition x"""
struct Projection <: ConstraintUpdate
    n::Integer # TODO should these be Int? Or do we parameterize the update types? Or does it not make a difference?
    proj::Union{ProjectedNormalization, Entrywise} #ProjectedNormalization
end

check(P::Projection, D::AbstractDecomposition) = check(P.proj, factor(D, P.n))

function (U::Projection)(x::T; kwargs...) where T
    n = U.n
    if checkfrozen(x, n)
        return x
    end
    U.proj(factor(x, n))
end

NNProjection(n) = Projection(n, nonnegative!)

struct SafeNNProjection <: ConstraintUpdate
    n::Integer
    backup::ProjectedNormalization
end

function (U::SafeNNProjection)(x::T; kwargs...) where T
    n = U.n
    if checkfrozen(x, n)
        return x
    end

    A = factor(x, n)
    @debug display(A)
    for a in U.backup.whats_normalized(A)
        if all(a .<= 0)
            # "a" would be projected to the origin so apply the backup projection
            U.backup.projection(a)
        else
            nonnegative!(a)
        end
    end
    @debug display(A)
    check(U, x) || error("Something went wrong with SafeNNProjection using the backup projection: $(U.backup)")
end

check(S::SafeNNProjection, D::AbstractDecomposition) = check(nonnegative!, factor(D, S.n)) && all(!iszero, S.backup.whats_normalized(factor(D, S.n)))

"""
    Rescale{T<:Union{Nothing,Missing,Function}} <: ConstraintUpdate
    Rescale(n, scale::ScaledNormalization, whats_rescaled::T)

Applies the scaled normalization `scale` to factor `n`, and tries to multiply the scaling of
factor `n` to other factors.

If `whats_rescaled=nothing`, then it will not rescale any other factor.

If `whats_rescaled=missing`, then it will try to evenly distribute the weight to all other
factors using the (N-1) root of each weight where N is the number of factors. If the weights
are not broadcastable, (e.g. you want to scale each row but each factor has a different number
of rows), will use the geometric mean of the weights as the single weight to distribute evenly
among the other factors.

If `typeof(whats_rescaled) <: Function`, will broadcast the weight to the output of calling
this function on the entire decomposition. For example,
    `whats_rescale = x -> eachcol(factor(x, 2))`
will rescale each column of the second factor of the decomposition.
"""
struct Rescale{T<:Union{Nothing,Missing,Function}} <: ConstraintUpdate
    n::Integer
    scale::ScaledNormalization
    whats_rescaled::T
end

check(S::Rescale, D::AbstractDecomposition) = check(S.scale, factor(D, S.n))

function (U::Rescale{<:Function})(x; kwargs...)
    # TODO possible have information about what gets rescaled with the `ScaledNormalization`.
    # Right now, the scaling is only applied to arrays, not decompositions, so the information
    # about where (`U.whats_rescaled`) and how (only multiplication (*) right now) the weight
    # from Fn gets canceled out is stored with the `Rescale` struct and not
    # the `ScaledNormalization`.
    Fn_scale = U.scale(factor(x, U.n))
    to_scale = U.whats_rescaled(x)
    to_scale .*= Fn_scale
end

(U::Rescale{Nothing})(x; kwargs...) = U.scale(factor(x, U.n))
function (U::Rescale{Missing})(x; skip_rescale=false, kwargs...)
    Fn_scale = U.scale(factor(x, U.n))
    x_factors = factors(x)
    N = length(x_factors) - 1

    # Nothing to rescale, so return here
    if N == 0 || skip_rescale # we keep this option so that we can always override the rescaling part, even after we've constructed the update
        return nothing
    end

    # Assume we want to evenly rescale all other factors by the Nth root of Fn_scale
    scale = geomean(Fn_scale)^(1/N) #TODO test this actually works
    for (i, A) in zip(eachfactorindex(x), x_factors)
        # skip over the factor we just updated
        if i == U.n
            continue
        end
        A .*= scale
    end
end

function (U::Rescale{Missing})(x::CPDecomposition; skip_rescale=false, kwargs...)
    Fn_scale = U.scale(factor(x, U.n))
    x_factors = factors(x)
    N = length(x_factors) - 1

    # Nothing to rescale, so return here
    if N == 0 || skip_rescale # we keep this option so that we can always override the rescaling part, even after we've constructed the update
        return nothing
    end

    # Assume we want to evenly rescale all other factors by the Nth root of Fn_scale
    scale = Fn_scale .^ (1/N)
    for (i, A) in zip(eachfactorindex(x), x_factors)
        # skip over the factor we just updated
        if i == U.n
            continue
        end
        A_whats_normalized = U.scale.whats_normalized(A)
        A_whats_normalized .*= scale
    end
end

struct MomentumUpdate <: AbstractUpdate
    n::Integer
    lipschitz::Function
    combine::Function # How to combine the momentum variable `ω` with a factor `a`
end

MomentumUpdate(n, lipschitz) = MomentumUpdate(n, lipschitz, (ω, a) -> ω * a)

"""
Makes a MomentumUpdate from an AbstractGradientDescent assuming the AbstractGradientDescent has a lipschitz step size
"""
function MomentumUpdate(GD::AbstractGradientDescent)
    n, step = GD.n, GD.step
    @assert typeof(step) <: LipschitzStep

    return MomentumUpdate(n, step.lipschitz)
end

function MomentumUpdate(GD::BlockGradientDescent)
    n, step, combine = GD.n, GD.step, GD.combine
    @assert typeof(step) <: LipschitzStep

    return MomentumUpdate(n, step.lipschitz, combine)
end

function (U::MomentumUpdate)(x::T; x_last::T, ω, δ, kwargs...) where T
    n = U.n
    if checkfrozen(x, n)
        return x
    end
    # TODO avoid redoing this lipschitz calculation and instead store the previous L
    # TODO generalize this momentum update to allow for other decaying momentums ω
    L = U.lipschitz(x; kwargs...)
    L_last = U.lipschitz(x_last; kwargs...)
    ω = min.(ω, δ .* .√(L_last/L))

    a, a_last = factor(x, n), factor(x_last, n)

    # Equivalent mathematically, but slightly less efficient ways of applying momentum
    # @. a = a + ω * (a - a_last)

    # @. a += ω * (a - a_last)

    # a .*= 1 + a
    # a .-= ω .* a_last

    a = U.combine(a, id + ω) # handle diagonal Lipschitz constants
    a .-= U.combine(a_last, ω)
end

struct BlockedUpdate <: AbstractUpdate
    updates::Vector{AbstractUpdate}
    # Note I want exactly AbstractUpdate[] since I want to push any type of AbstractUpdate
    # like MomentumUpdate or another BlockedUpdate, even if not already present.
    # This means it cannot be Vector{<:AbstractUpdate} since a BlockedUpdate with only
    # GradientDescent would give a GradientDescent[] and we couldn't push a MomentumUpdate.
    # And it cannot be AbstractVector{AbstractUpdate} since we may not be able to insert!/push!
    # into other AbstractVectors like Views.
end

#convert(, x::AbstractVector{<:AbstractUpdate})

# Wrappers to allow multiple args, or a tuple input
BlockedUpdate(x::Tuple) = BlockedUpdate(x...)
BlockedUpdate(x...) = BlockedUpdate(vcat(x...))

function Base.getproperty(U::BlockedUpdate, sym::Symbol)
    if sym === :n
        if allequal(getproperty.(updates(U), :n))
            return U[begin].n # can safely say U is just multiple updates on factor n
        else
            @debug "BlockedUpdate contains updates to multiple factor indices:\n$(getproperty.(updates(U), :n))"
            return nothing #throw(ErrorException("BlockedUpdate contains updates to multiple factor indices:\n$(getproperty.(updates(U), :n))"))
        end
    else # fallback to other fields
        return getfield(U, sym)
    end
end

updates(U::BlockedUpdate) = U.updates

# Forward methods to Vector so BlockedUpdate can behave like a Vector
Base.getindex(U::BlockedUpdate, i::Int) = getindex(updates(U), i)
Base.getindex(U::BlockedUpdate, I::Vararg{Int}) = getindex(updates(U), I...)
Base.getindex(U::BlockedUpdate, I) = getindex(updates(U), I) # catch all
Base.firstindex(U::BlockedUpdate) = firstindex(updates(U))
Base.lastindex(U::BlockedUpdate) = lastindex(updates(U))
Base.keys(U::BlockedUpdate) = keys(updates(U))
Base.length(U::BlockedUpdate) = length(updates(U))
Base.iterate(U::BlockedUpdate, state=1) = state > length(U) ? nothing : (U[state], state+1)
Base.filter(f, U::BlockedUpdate) = BlockedUpdate(filter(f, updates(U)))

# for blocked update of ConstraintUpdate
check(U::BlockedUpdate, D::AbstractDecomposition) = all(u -> check(u, D), U)

function Base.show(io::IO, x::BlockedUpdate)
    println(io, typeof(x), "(")
    out = join(split(join(string.(updates(x)), "\n"), "\n"), "\n    ")

    println(io, "    ", out)
    print(io, ")")
end

Base.show(io::IO, ::MIME"text/plain", x::BlockedUpdate) = show(io, x)

function (U::BlockedUpdate)(x::T; recursive_random_order::Bool=false, random_order::Bool=recursive_random_order, kwargs...) where T
    U_updates = updates(U)
    if random_order
        order = shuffle(eachindex(U_updates))
        U_updates = U_updates[order] # not using a view since we need to acess elements in a random order
                                   # https://docs.julialang.org/en/v1/manual/performance-tips/#Copying-data-is-not-always-bad
    end

    for update! in U_updates
        update!(x; recursive_random_order, kwargs...) # note random_order does not get passed down
    end
end

function add_momentum!(U::BlockedUpdate)
    # Find all the GradientDescent updates
    U_updates = updates(U)
    indexes = findall(u -> typeof(u) <: AbstractGradientDescent, U_updates)

    # insert MomentumUpdates before each GradientDescent
    # do this in reverse order so "i" correctly indexes a GradientDescent
    # as we mutate updates
    for i in reverse(indexes)
        insert!(U_updates, i, MomentumUpdate(U_updates[i]))
    end
end

"""
    smart_insert!(U::BlockedUpdate, V::AbstractUpdate)

Tries to insert V into U after the last matching update in U. A "matching update" means
it updates the same factor/block n.
See [`smart_interlace!`](@ref)
"""
function smart_insert!(U::BlockedUpdate, V::AbstractUpdate)
    U_updates = updates(U)
    i = findlast(u -> u.n == V.n, U_updates)

    # insert the other update immediately after
    # or if there is no update, push it to the end
    isnothing(i) ? push!(U_updates, V) : insert!(U_updates, i+1, V)
end

"""
    smart_interlace!(U::BlockedUpdate, V)

`smart_insert!`s each update in V, into U.
See [`smart_insert!`](@ref)
"""
function smart_interlace!(U::BlockedUpdate, other_updates)
    for V in other_updates
        smart_insert!(U::BlockedUpdate, V::AbstractUpdate)
    end
end

smart_interlace!(U::BlockedUpdate, V::BlockedUpdate) = smart_interlace!(U::BlockedUpdate, updates(V))

"""
    group_by_factor(blockedupdate::BlockedUpdate)

Groups updates according to the factor they operate on.

If blockedupdate contains other `BlockedUpdate`s, the inner updates are grouped when they
all operate on the same factor.

Updates which do not have an assigned factor are grouped together.

The order which these groups appear in the output follows the same order as the first
appearence of each unique factor that is operated on.
"""
function group_by_factor(blockedupdate::BlockedUpdate)
    factor_labels = unique(getproperty(U, :n) for U in blockedupdate)
    updates_by_factor = [filter(U -> U.n == n, blockedupdate) for n in factor_labels]
    return BlockedUpdate(updates_by_factor)
end
