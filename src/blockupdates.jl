"""
Mid level code that combines constraints with block updates to be used on an AbstractDecomposition
"""

abstract type AbstractUpdate{T<:AbstractDecomposition} <: Function end

struct GenericUpdate{T} <: AbstractUpdate{T}
    f::Function
end

(U::GenericUpdate{T})(x::T) where T = (U.f)(x)

#=
struct ProxGradientUpdate{T} <: AbstractUpdate{T}
    gradientstep::Function
    prox::AbstractConstraint
end

(U::ProxGradientUpdate{T})(x::T) = (U.prox ∘ U.gradientstep)(x)
=#

"""Perform a GradientUpdate on the nth factor of an Abstract Decomposition x"""
struct GradientUpdate{T} <: AbstractUpdate{T}
    gradientstep::Function
    n::Integer
end

function (U::GradientUpdate{T})(x::T) where T
    n = U.n
    if isfrozen(x, n)
        return x
    end
    U.gradientstep(x)
end

"""Perform a projected gradient update on the nth factor of an Abstract Decomposition x"""
struct ProjGradientUpdate{T} <: AbstractUpdate{T}
    gradientstep::Function
    proj::ProjectedNormalization
    n::Integer
end

function (U::ProjGradientUpdate{T})(x::T) where T
    n = U.n
    if isfrozen(x, n)
        return x
    end
    U.gradientstep(x)
    U.proj(factor(x, n))
end

"""
Perform a nonnegative gradient update on the nth factor of an Abstract Decomposition x.
See [`ProjGradientUpdate`](@ref).
"""
struct NNGradientUpdate{T} <: AbstractUpdate{T}
    gradientstep::Function
    n::Integer
end

function (U::NNGradientUpdate{T})(x::T) where T
    n = U.n
    if isfrozen(x, n)
        return x
    end
    U.gradientstep(x)
    nnegative!(factor(x, n))
end

struct ScaledNNGradientUpdate{T} <: AbstractUpdate{T}
    gradientstep::Function
    scale::ScaledNormalization
    whats_rescaled::Function
    n::Integer
end

function (U::ScaledNNGradientUpdate{T})(x::T) where T
    n = U.n
    if isfrozen(x, n)
        return x
    end
    Fn = factor(x, n)

    U.gradientstep(x)
    nnegative!(Fn)

    # TODO possible have information about what gets rescaled withthe `ScaledNormalization`.
    # Right now, the scaling is only applied to arrays, not decompositions, so the information
    # about where (`U.whats_rescaled`) and how (only multiplication (*) right now) the weight
    # from Fn gets canceled out is stored with the `ScaledNNGradientUpdate` struct and not
    # the `ScaledNormalization`.
    Fn_scale = U.scale(Fn)
    to_scale = U.whats_rescaled(x)
    to_scale .*= Fn_scale
end

struct BlockedUpdate{T} <: AbstractUpdate{T}
    updates::NTuple{N, AbstractUpdate{T}} where N
end

#BlockedUpdate(updates::NTuple{N, AbstractUpdate{T}}) where {T, N} = BlockedUpdate{T}(updates)

function (U::BlockedUpdate{T})(x::T; random_order::Bool=false) where T
    if random_order
        order = shuffle(eachindex(U.updates))
        updates = U.updates[order]
        for update! in updates
            update!(x)
        end
    else
        for update! in U.updates
            update!(x)
        end
    end
end

##################################################

function make_gradstep_core(Y::AbstractArray; stepsize=:lipshitz, kwargs...)
    function gradstep_core!(T::Tucker1; kwargs...)
        (C, A) = factors(T)
        AA = A'A
        YA = Y×₁A'
        grad = C×₁AA - YA # TODO define multiplication generaly
        stepsize == :lipshitz ? step=1/opnorm(AA) : step=stepsize # TODO make a function that takes a symbol for a type of stepsize and calculates the step
        @. C -= step * grad
        return C
    end
    return gradstep_core!
end

function make_gradstep_matrix(Y::AbstractArray; stepsize=:lipshitz, kwargs...)
    function gradstep_matrix!(T::Tucker1; kwargs...)
        (C, A) = factors(T)
        CC = slicewise_dot(C, C)
        YC = slicewise_dot(Y, C)
        grad = A*CC - YC
        stepsize == :lipshitz ? step=1/opnorm(CC) : step=stepsize # TODO make a function that takes a symbol for a type of stepsize and calculates the step
        @. A -= step * grad
        return C
    end
    return gradstep_matrix!
end

function block_gradient_decent(T::Tucker1, Y::AbstractArray; kwargs...)
    size(T) == size(Y) || ArgumentError("Size of decomposition $(size(T)) does not match size of the data $(size(Y))")
    gradstep_core! = make_gradstep_core(Y; kwargs...)
    gradstep_matrix! = make_gradstep_matrix(Y; kwargs...)
    block_updates = (
        GradientUpdate{Tucker1}(gradstep_core!, 1),
        GradientUpdate{Tucker1}(gradstep_matrix!, 1),
    )
    return BlockedUpdate(block_updates)
end

function nn_block_gradient_decent(T::Tucker1, Y::AbstractArray; kwargs...)
    size(T) == size(Y) || ArgumentError("Size of decomposition $(size(T)) does not match size of the data $(size(Y))")
    gradstep_core! = make_gradstep_core(Y; kwargs...)
    gradstep_matrix! = make_gradstep_matrix(Y; kwargs...)
    block_updates = (
        NNGradientUpdate{Tucker1}(gradstep_core!, 1),
        NNGradientUpdate{Tucker1}(gradstep_matrix!, 1),
    )
    return BlockedUpdate(block_updates)
end

function scaled_nn_block_gradient_decent(T::Tucker1, Y::AbstractArray; scale, whats_rescaled, kwargs...)
    size(T) == size(Y) || ArgumentError("Size of decomposition $(size(T)) does not match size of the data $(size(Y))")
    gradstep_core! = make_gradstep_core(Y; kwargs...)
    gradstep_matrix! = make_gradstep_matrix(Y; kwargs...)
    block_updates = (
        ScaledNNGradientUpdate{Tucker1}(gradstep_core!, scale, whats_rescaled, 1),
        NNGradientUpdate{Tucker1}(gradstep_matrix!, 1),
    )
    return BlockedUpdate(block_updates)
end

function proj_nn_block_gradient_decent(T::Tucker1, Y::AbstractArray; proj, kwargs...)
    size(T) == size(Y) || ArgumentError("Size of decomposition $(size(T)) does not match size of the data $(size(Y))")
    gradstep_core! = make_gradstep_core(Y; kwargs...)
    gradstep_matrix! = make_gradstep_matrix(Y; kwargs...)
    block_updates = (
        ProjGradientUpdate{Tucker1}(gradstep_core!, proj, 1),
        NNGradientUpdate{Tucker1}(gradstep_matrix!, 1),
    )
    return BlockedUpdate(block_updates)
end

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
