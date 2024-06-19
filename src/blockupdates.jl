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
    @. a -= s*g
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

    elseif n==1 # the matrix is the zeroth factor
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

"""Perform a projected gradient update on the nth factor of an Abstract Decomposition x"""
struct Projection <: AbstractUpdate
    n::Integer
    proj::AbstractConstraint #ProjectedNormalization
end

function (U::Projection)(x::T; kwargs...) where T
    n = U.n
    if checkfrozen(x, n)
        return x
    end
    U.proj(factor(x, n))
end

NNProjection(n) = Projection(n, nnegative!)

struct Rescale <: AbstractUpdate
    n::Integer
    scale::ScaledNormalization
    whats_rescaled::Function
end

function (U::Rescale)(x; kwargs...)
    # TODO possible have information about what gets rescaled withthe `ScaledNormalization`.
    # Right now, the scaling is only applied to arrays, not decompositions, so the information
    # about where (`U.whats_rescaled`) and how (only multiplication (*) right now) the weight
    # from Fn gets canceled out is stored with the `Rescale` struct and not
    # the `ScaledNormalization`.
    Fn_scale = U.scale(Fn)
    to_scale = U.whats_rescaled(x)
    to_scale .*= Fn_scale
end

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
    @. a += ω * (a - a_last)
end

struct BlockedUpdate <: AbstractUpdate
    updates::NTuple{N, AbstractUpdate} where N
end

#BlockedUpdate(updates::NTuple{N, AbstractUpdate{T}}) where {T, N} = BlockedUpdate{T}(updates)

function (U::BlockedUpdate)(x::T; random_order::Bool=false, kwargs...) where T
    if random_order
        order = shuffle(eachindex(U.updates))
        updates = U.updates[order]
        for update! in updates
            update!(x; kwargs...)
        end
    else
        for update! in U.updates
            update!(x; kwargs...)
        end
    end
end

##################################################

# Attempt 3
#=
function make_momentum_gradstep_matrix(Y::AbstractArray; kwargs...)
    # This momentum update only works for lipshitz stepsize
    # I leave calcstep as an option so we error if a different calcstep was requested
    function momentum_gradstep_matrix!(T::Tucker1;
            calcstep=LipshitzStep()::LipshitzStep, δ=0, ω=0, kwargs...)
        (C, A) = factors(T)
        CC = slicewise_dot(C, C)
        YC = slicewise_dot(Y, C)

        # Momentum
        L = opnorm(CC)
        ω = min(ω, δ*sqrt(LA/L))
        A .+= momentum

        # Gradient
        grad = A*CC - YC
        step=calcstep(L)
        @. A -= step * grad
        return C
    end
    return momentum_gradstep_matrix!
end

# Attempt 2
function make_momentum_gradstep_matrix(Y::AbstractArray; kwargs...)
    # This momentum update only works for lipshitz stepsize
    # I leave this as an option so we error if a different calcstep was requested
    function momentum_gradstep_matrix!(T::Tucker1;
            calcstep=LipshitzStep()::LipshitzStep, δ=0, ω=0, kwargs...)
        (C, A) = factors(T)
        CC = slicewise_dot(C, C)
        YC = slicewise_dot(Y, C)

        # Momentum
        L = opnorm(CC)
        ω = min(ω, δ*sqrt(LA/L))
        A .+= momentum

        # Gradient
        grad = A*CC - YC
        step=calcstep(L)
        @. A -= step * grad
        return C
    end
    return momentum_gradstep_matrix!
end

# Attempt 1
function make_momentum_gradstep_matrix(Y::AbstractArray; kwargs...)
    function momentum_gradstep_matrix!(T::Tucker1; momentum=0, stepsize=:lipshitz, kwargs...)
        (C, A) = factors(T)
        iszero(momentum) ? nothing : A .+= momentum
        CC = slicewise_dot(C, C)
        YC = slicewise_dot(Y, C)
        grad = A*CC - YC
        stepsize == :lipshitz ? step=1/opnorm(CC) : step=stepsize # TODO make a function that takes a symbol for a type of stepsize and calculates the step
        @. A -= step * grad
        return A
    end
    return momentum_gradstep_matrix!
end
=#
#=
function make_gradstep_core(Y::AbstractArray; kwargs...)
    function gradstep_core!(T::Tucker1; stepsize=:lipshitz, kwargs...)
        (C, A) = factors(T)
        AA = A'A
        YA = Y×₁A'
        grad = C×₁AA - YA # TODO define multiplication generaly
        stepsize == :lipshitz ? step=1/opnorm(AA) : step=stepsize # step = calcstep(opnorm(AA)) # TODO make a function that takes a symbol for a type of stepsize and calculates the step
        @. C -= step * grad
        return C
    end
    return gradstep_core!
end

function make_gradstep_matrix(Y::AbstractArray; kwargs...)
    function gradstep_matrix!(T::Tucker1; stepsize=:lipshitz, kwargs...)
        (C, A) = factors(T)
        CC = slicewise_dot(C, C)
        YC = slicewise_dot(Y, C)
        grad = A*CC - YC
        stepsize == :lipshitz ? step=1/opnorm(CC) : step=stepsize # TODO make a function that takes a symbol for a type of stepsize and calculates the step
        @. A -= step * grad
        return A
    end
    return gradstep_matrix!
end

function block_gradient_decent(T::Tucker1, Y::AbstractArray; kwargs...)
    size(T) == size(Y) || ArgumentError("Size of decomposition $(size(T)) does not match size of the data $(size(Y))")
    gradstep_core! = make_gradstep_core(Y; kwargs...)
    gradstep_matrix! = make_gradstep_matrix(Y; kwargs...)
    block_updates = (
        GradientUpdate{Tucker1}(gradstep_core!, 1),
        GradientUpdate{Tucker1}(gradstep_matrix!, 2),
    )
    return BlockedUpdate(block_updates)
end

function nn_block_gradient_decent(T::Tucker1, Y::AbstractArray; kwargs...)
    size(T) == size(Y) || ArgumentError("Size of decomposition $(size(T)) does not match size of the data $(size(Y))")
    gradstep_core! = make_gradstep_core(Y; kwargs...)
    gradstep_matrix! = make_gradstep_matrix(Y; kwargs...)
    block_updates = (
        NNGradientUpdate{Tucker1}(gradstep_core!, 1),
        NNGradientUpdate{Tucker1}(gradstep_matrix!, 2),
    )
    return BlockedUpdate(block_updates)
end

function scaled_nn_block_gradient_decent(T::Tucker1, Y::AbstractArray; core_constraint, whats_rescaled, kwargs...)
    size(T) == size(Y) || ArgumentError("Size of decomposition $(size(T)) does not match size of the data $(size(Y))")
    gradstep_core! = make_gradstep_core(Y; kwargs...)
    gradstep_matrix! = make_gradstep_matrix(Y; kwargs...)
    block_updates = (
        ScaledNNGradientUpdate{Tucker1}(gradstep_core!, core_constraint, whats_rescaled, 1),
        NNGradientUpdate{Tucker1}(gradstep_matrix!, 2),
    )
    return BlockedUpdate(block_updates)
end

function proj_nn_block_gradient_decent(T::Tucker1, Y::AbstractArray; proj, kwargs...)
    size(T) == size(Y) || ArgumentError("Size of decomposition $(size(T)) does not match size of the data $(size(Y))")
    gradstep_core! = make_gradstep_core(Y; kwargs...)
    gradstep_matrix! = make_gradstep_matrix(Y; kwargs...)
    block_updates = (
        ProjGradientUpdate{Tucker1}(gradstep_core!, proj, 1),
        NNGradientUpdate{Tucker1}(gradstep_matrix!, 2),
    )
    return BlockedUpdate(block_updates)
end

function momentum_scaled_nn_block_gradient_decent(T::Tucker1, Y::AbstractArray; core_constraint, whats_rescaled, kwargs...)
    size(T) == size(Y) || ArgumentError("Size of decomposition $(size(T)) does not match size of the data $(size(Y))")
    gradstep_core! = make_gradstep_core(Y; kwargs...)
    gradstep_matrix! = make_gradstep_matrix(Y; kwargs...)
    block_updates = (
        MomentumUpdate{Tucker1}(1), #need different ω's for these two momentum updates...
        ScaledNNGradientUpdate{Tucker1}(gradstep_core!, core_constraint, whats_rescaled, 1),
        MomentumUpdate{Tucker1}(2),
        NNGradientUpdate{Tucker1}(gradstep_matrix!, 2),
    )
    return BlockedUpdate(block_updates)
end

=#
########################################################################################


#=
forwarded_functions = (
    # AbstractDecomposition Interface
    :array,
    :factors,
    :contractions,
    :rankof,
    # AbstractArray Interface
    #:(Base.ndims),
    :(Base.size),
    :(Base.getindex),
)
for f in forwarded_functions
    @eval ($f)(BUD::BlockUpdatedDecomposition) = ($f)(decomposition(BUD))
end
=#
