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

#(U::ProjGradientUpdate{T})(x::T) where T = (U.proj ∘ U.gradientstep)(x)

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

    (U.gradientstep)(x)
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

function block_gradient_decent(T::Tucker1, Y::AbstractArray)
    size(T) == size(Y) || ArgumentError("Size of decomposition $(size(T)) does not match size of the data $(size(Y))")

    function gradstep_core!(T::Tucker1; stepsize=:lipshitz, kwargs...)
        (C, A) = factors(T)
        AA = A'A
        YA = Y ×₁A'
        grad = C×₁AA - YA # TODO define multiplication generaly
        stepsize == :lipshitz ? step=1/opnorm(AA) : step=stepsize # TODO make a function that takes a symbol for a type of stepsize and calculates the step
        @. C -= step * grad
        return C
    end

    function gradstep_matrix!(T::Tucker1; stepsize=:lipshitz, kwargs...)
        (C, A) = factors(T)
        CC = slicewise_dot(C, C)
        YC = slicewise_dot(Y, C)
        grad = A*CC - YC
        stepsize == :lipshitz ? step=1/opnorm(CC) : step=stepsize # TODO make a function that takes a symbol for a type of stepsize and calculates the step
        @. A -= step * grad
        return C
    end

    block_updates = (GradientUpdate{Tucker1}(gradstep_core!, 1), GradientUpdate{Tucker1}(gradstep_matrix!, 1))
    return BlockedUpdate(block_updates)
end

#=
"""Main type holding information about how to update each block in an AbstractDecomposition"""
struct BlockedUpdate <: AbstractUpdate
	D::AbstractDecomposition{T, N}
	updates::NTuple{M, Function} where M
	function BlockUpdatedDecomposition{T, N}(D, block_updates) where {T, N}
		n = nfactors(D)
		m = length(block_updates)
		n == m ||
            throw(ArgumentError("Number of factors $n does not match the number of block_updates $m"))
        new{T, N}(D, block_updates)
	end
end

# Constructor
# This is needed since we need to pass the type information from D, to the constructor
# `BlockUpdatedDecomposition{T, N}`
function BlockUpdatedDecomposition(D::AbstractDecomposition{T, N}, block_updates) where {T, N}
    return BlockUpdatedDecomposition{T, N}(D, block_updates)
end

# BlockUpdatedDecomposition Interface
decomposition(BUD::BlockUpdatedDecomposition) = BUD.D
function updates(BUD::BlockUpdatedDecomposition; rand_order=false, kwargs...)
    updates_tuple = BUD.updates
    return rand_order ? updates_tuple[randperm(length(updates_tuple))] : updates_tuple
end
function update!(BUD::BlockUpdatedDecomposition; kwargs...)
    D = decomposition(BUD)
    for block_update! in updates(BUD; kwargs...)
        block_update!(D)
    end
    return BUD
end
#each_update_factor(G::BlockUpdatedDecomposition) = zip(updates(G),factors(G))

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

function least_square_updates(T::Tucker1, Y::AbstractArray)
    size(T) == size(Y) || ArgumentError("Size of decomposition $(size(T)) does not match size of the data $(size(Y))")

    function update_core!(T::Tucker1; stepsize=:lipshitz, kwargs...)
        (C, A) = factors(T)
        AA = A'A
        YA = Y ×₁A'
        grad = C×₁AA - YA # TODO define multiplication generaly
        stepsize == :lipshitz ? step=1/opnorm(AA) : step=stepsize # TODO make a function that takes a symbol for a type of stepsize and calculates the step
        @. C -= step * grad
        return C
    end

    function update_matrix!(T::Tucker1; stepsize=:lipshitz, kwargs...)
        (C, A) = factors(T)
        CC = slicewise_dot(C, C)
        YC = slicewise_dot(Y, C)
        grad = A*CC - YC
        stepsize == :lipshitz ? step=1/opnorm(CC) : step=stepsize # TODO make a function that takes a symbol for a type of stepsize and calculates the step
        @. A -= step * grad
        return C
    end

    block_updates = (update_core!, update_matrix!)
    return BlockUpdatedDecomposition(T, block_updates)
end
=#
