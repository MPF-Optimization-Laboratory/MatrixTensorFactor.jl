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

struct LipshitzStep <: AbstractStep
    lipshitz::Function
end

function (step::LipshitzStep)(x; kwargs...)
    L = step.lipshitz(x)
    return 1/L
end
#LipshitzStep(L::Real) = 1/L

function make_lipshitz(T::Tucker1, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    if n==0 # the core is the zeroth factor
        function lipshitz0(T::Tucker1; kwargs...)
            A = matrix_factor(T, 1)
            return opnorm(A'A)
        end
        return lipshitz0

    elseif n==1 # the matrix is the zeroth factor
        function lipshitz1(T::Tucker1; kwargs...)
            C = core(T)
            return opnorm(slicewise_dot(C, C))
        end
        return lipshitz1

    else
        error("No $(n)th factor in Tucker1")
    end
end

function make_lipshitz(T::Tucker, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n==0 # the core is the zeroth factor
        function lipshitz_core(T::AbstractTucker; kwargs...)
            #matricies = matrix_factors(T)
            #gram_matricies = map(A -> A'A, matricies)
            #return prod(opnorm.(gram_matricies))
            return prod(A -> opnorm(A'A), matrix_factors(T))
        end
        return lipshitz_core

    elseif n in 1:N # the matrix is the zeroth factor
        function lipshitz_matrix(T::AbstractTucker; kwargs...)
            matricies = matrix_factors(T)
            TExcludeAn = tuckerproduct(core(T), getnotindex(matricies, n); exclude=n)
            return opnorm(slicewise_dot(TExcludeAn, TExcludeAn; dims=n))
        end
        return lipshitz_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end

function make_lipshitz(T::CPDecomposition, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n in 1:N # the matrix is the zeroth factor
        function lipshitz_matrix(T::AbstractTucker; kwargs...)
            matricies = matrix_factors(T)
            TExcludeAn = tuckerproduct(core(T), getnotindex(matricies, n); exclude=n) # TODO optimize this to avoid making the super diagonal core
            return opnorm(slicewise_dot(TExcludeAn, TExcludeAn; dims=n))
        end
        return lipshitz_matrix

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
# so I will skip implimenting it for now, but may want to add that once
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
        if p == :n
            push!(data, getproperty(x, p))
        elseif p in (:step, :proj)
            push!(data, typeof(getproperty(x, p)))
        else
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

#=
struct ProxGradientUpdate{T} <: AbstractUpdate{T}
    gradientstep::Function
    prox::AbstractConstraint
end

(U::ProxGradientUpdate{T})(x::T) = (U.prox ∘ U.gradientstep)(x)
=#

function checkfrozen(x, n)
    frozen = isfrozen(x, n)
    if frozen
        @warn "Factor $n is frozen, skipping its update."
    end
    return frozen
end

"""
Perform a Gradient decent step on the nth factor of an Abstract Decomposition x

The n is only to keep track of the factor that gets updated, and to check if a frozen
factor was requested to be updated.
"""
struct GradientDescent <: AbstractUpdate
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
    error("Gradient not implimented for ", typeof(D), " with ", typeof(objective), " objective")
end

# Using this patter of inputs so that gradients for a generic decomposition could be calculated
# with auto diff by looking at the gradient of the function objective(D, Y) with respect to the nth factor in D
function make_gradient(T::Tucker1, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    if n==0 # the core is the zeroth factor
        function gradient0(T::Tucker1; kwargs...)
            (C, A) = factors(T)
            AA = A'A
            YA = Y×₁A'
            grad = C×₁AA - YA # TODO define multiplication generaly
            return grad
        end
        return gradient0

    elseif n==1 # the matrix is the first factor
        function gradient1(T::Tucker1; kwargs...)
            (C, A) = factors(T)
            CC = slicewise_dot(C, C)
            YC = slicewise_dot(Y, C)
            grad = A*CC - YC
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
        function gradient_core(T::AbstractTucker; kwargs...)
            C = core(T)
            matricies = matrix_factors(T)
            gram_matricies = map(A -> A'A, matricies) # gram matricies AA = A'A, BB = B'B...
            YAB = tuckerproduct(Y, adjoint.(matricies)) # Y ×₁ A' ×₂ B' ...
            grad = tuckerproduct(C, gram_matricies) - YAB
            return grad
        end
        return gradient_core

    elseif n in 1:N # the matrix factors start at m=1
        function gradient_matrix(T::AbstractTucker; kwargs...)
            matricies = matrix_factors(T)
            TExcludeAn = tuckerproduct(core(T), getnotindex(matricies, n); exclude=n)
            An = factor(T, n)
            grad = An*slicewise_dot(TExcludeAn, TExcludeAn; dims=n) - slicewise_dot(Y, TExcludeAn; dims=n)
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
        function gradient_matrix(T::AbstractTucker; kwargs...)
            matricies = matrix_factors(T)
            TExcludeAn = tuckerproduct(core(T), getnotindex(matricies, n); exclude=n)
            An = factor(T, n)
            grad = An*slicewise_dot(TExcludeAn, TExcludeAn; dims=n) - slicewise_dot(Y, TExcludeAn; dims=n)
            return grad
        end
        return gradient_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end

abstract type ConstraintUpdate <: AbstractUpdate end

"""
    ConstraintUpdate(n, constraint)

Converts an AbstractConstraint to a ConstraintUpdate on the factor n
"""
ConstraintUpdate(n, constraint::AbstractConstraint; kwargs...) = error("converting $(typeof(constraint)) to a ConstraintUpdate is not yet supported")
ConstraintUpdate(n, constraint::ProjectedNormalization; kwargs...) = Projection(n, constraint)
ConstraintUpdate(n, constraint::ScaledNormalization; whats_rescaled=missing, kwargs...) = Rescale(n, constraint, whats_rescaled)
ConstraintUpdate(n, constraint::EntryWise; kwargs...) = Projection(n, constraint)
ConstraintUpdate(n, constraint::ComposedConstraint; kwargs...) = BlockedUpdate(ConstraintUpdate(n, constraint.inner; kwargs...), ConstraintUpdate(n, constraint.outer; kwargs...)) # note we apply inner constraint first

check(_::ConstraintUpdate, _::AbstractDecomposition) = error("checking $(typeof(constraint)) is not yet supported")

"""Perform a projected gradient update on the nth factor of an Abstract Decomposition x"""
struct Projection <: ConstraintUpdate
    n::Integer
    proj::AbstractConstraint #ProjectedNormalization
end

check(P::Projection, D::AbstractDecomposition) = check(P.proj, factor(D, P.n))

function (U::Projection)(x::T; kwargs...) where T
    n = U.n
    if checkfrozen(x, n)
        return x
    end
    U.proj(factor(x, n))
end

NNProjection(n) = Projection(n, nnegative!)

struct Rescale{T<:Union{Nothing,Missing,Function}} <: ConstraintUpdate
    n::Integer
    scale::ScaledNormalization
    whats_rescaled::T
end

check(S::Rescale, D::AbstractDecomposition) = check(S.scale, factor(D, S.n))

function (U::Rescale{<:Function})(x; kwargs...)
    # TODO possible have information about what gets rescaled withthe `ScaledNormalization`.
    # Right now, the scaling is only applied to arrays, not decompositions, so the information
    # about where (`U.whats_rescaled`) and how (only multiplication (*) right now) the weight
    # from Fn gets canceled out is stored with the `Rescale` struct and not
    # the `ScaledNormalization`.
    Fn_scale = U.scale(factor(x, U.n))
    to_scale = U.whats_rescaled(x)
    to_scale .*= Fn_scale
end

(U::Rescale{Nothing})(x; kwargs...) = U.scale(factor(x, U.n))
function (U::Rescale{Missing})(x; kwargs...)
    Fn_scale = U.scale(factor(x, U.n))
    x_factors = factors(x)
    N = length(x_factors) - 1

    # Nothing to rescale, so return here
    if N == 0
        return nothing
    end

    # Assume we want to evenly rescale all other factors by the Nth root of Fn_scale
    scale = geomean(Fn_scale)^(1/N)
    for (i, A) in zip(eachfactorindex(x), x_factors)
        # skip over the factor we just updated
        if i == U.n
            continue
        end
        A .*= scale
    end
end

function (U::Rescale{Missing})(x::CPDecomposition; kwargs...)
    Fn_scale = U.scale(factor(x, U.n))
    x_factors = factors(x)
    N = length(x_factors) - 1

    # Nothing to rescale, so return here
    if N == 0
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
#=
function Base.show(io::IO, ::MIME"text/plain", x::GradientDescent)
    print(io, typeof(x))
    print(io, "(", x.n, ", ...)")
end

function Base.show(io::IO, ::MIME"text/plain", x::ConstraintUpdate)
    print(io, typeof(x))
    print(io, "(", x.n, ", ...)")
end
=#

#=
struct MomentumUpdate <: AbstractUpdate
    n::Integer
    momentum::Function
end

function (U::MomentumUpdate)(x::T; x_last::T, kwargs...) where T
    n = U.n
    if checkfrozen(x, n)
        return x
    end
    ω = U.momentum(x; kwargs...)
    a, a_last = factor(x, n), factor(x_last, n)
    @. a += ω * (a - a_last)
end

function make_momentum(_::Tucker1, n::Integer, Y::AbstractArray; kwargs...)
    function momentum(T::Tucker1; ω, δ, lipshitz, L_last, kwargs...)
        L = lipshitz(T)
        return min(ω, δ * √(L_last/L)) # Safeguarded momentum step
    end
    return momentum
end
=#

struct MomentumUpdate <: AbstractUpdate
    n::Integer
    lipshitz::Function
end

"""
Makes a MomentumUpdate from a GradientDescent assuming the GradientDescent has a lipshitz step size
"""
function MomentumUpdate(GD::GradientDescent)
    n, step = GD.n, GD.step
    @assert typeof(step) <: LipshitzStep

    return MomentumUpdate(n, step.lipshitz)
end

function (U::MomentumUpdate)(x::T; x_last::T, ω, δ, kwargs...) where T
    n = U.n
    if checkfrozen(x, n)
        return x
    end
    # TODO avoid redoing this lipshitz calculation and instead store the previous L
    # TODO generalize this momentum update to allow for other decaying momentums ω
    L = U.lipshitz(x; kwargs...)
    L_last = U.lipshitz(x_last; kwargs...)
    ω = min(ω, δ * √(L_last/L))

    a, a_last = factor(x, n), factor(x_last, n)
    # @. a = a + ω * (a - a_last)
    # @. a += ω * (a - a_last)
    a .*= 1 + ω
    a .-= ω .* a_last
end
#=
function Base.show(io::IO, ::MIME"text/plain", x::MomentumUpdate)
    print(io, typeof(x))
    print(io, "(", x.n, ", ...)")
end
=#
struct BlockedUpdate <: AbstractUpdate
    updates::Vector{AbstractUpdate}
    # Note I want exactly AbstractUpdate[] since I want to push any type of AbstractUpdate
    # like MomentumUpdate or another BlockedUpdate, even if not already present.
    # This means it cannot be Vector{<:AbstractUpdate} since a BlockedUpdate with only
    # GradientDescent would give a GradientDescent[] and we couldnt push a MomentumUpdate.
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
            return throw(ErrorException("BlockedUpdate contains updates to multiple factor indicies:\n$(getproperty.(updates(U), :n))"))
        end
    else # fallback to other fields
        return getfield(U, sym)
    end
end

updates(U::BlockedUpdate) = U.updates

# Forward methods to Vector so BlockedUpdate can behave like a Vector
Base.getindex(U::BlockedUpdate, i::Int) = getindex(updates(U), i)
Base.getindex(U::BlockedUpdate, I::Vararg{Int}) = getindex(updates(U), I...)
Base.firstindex(U::BlockedUpdate) = firstindex(updates(U))
Base.lastindex(U::BlockedUpdate) = lastindex(updates(U))
Base.length(U::BlockedUpdate) = length(updates(U))
Base.iterate(U::BlockedUpdate, state=1) = state > length(U) ? nothing : (U[state], state+1)

function Base.show(io::IO, ::MIME"text/plain", x::BlockedUpdate)
    println(io, typeof(x), "(")
    for u in x
        println(io, "    ", u)
    end
    print(io, ")")
end

function (U::BlockedUpdate)(x::T; random_order::Bool=false, kwargs...) where T
    U_updates = updates(U)
    if random_order
        order = shuffle(eachindex(U_updates))
        U_updates = U_updates[order] # not using a view since we need to acess elements in a random order
                                   # https://docs.julialang.org/en/v1/manual/performance-tips/#Copying-data-is-not-always-bad
    end

    for update! in U_updates
        update!(x; kwargs...)
    end
end

function add_momentum!(U::BlockedUpdate)
    # Find all the GradientDescent updates
    U_updates = updates(U)
    indexes = findall(u -> typeof(u) <: GradientDescent, U_updates)

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
