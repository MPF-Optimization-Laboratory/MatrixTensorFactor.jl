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
    F = factors(x)
    (U.gradientstep)(F[n])
end

"""Perform a NNGradientUpdate on the nth factor of an Abstract Decomposition x"""
struct NNGradientUpdate{T} <: AbstractUpdate{T}
    gradientstep::Function
    n::Integer
end

function (U::NNGradientUpdate{T})(x::T) where T
    n = U.n
    if isfrozen(x, n)
        return x
    end
    F = factors(x)
    Fn = F[n]
    (U.gradientstep)(Fn)
    nnegative!(Fn)
end


struct ScaledNNGradientUpdate{T} <: AbstractUpdate{T}
    gradientstep::Function
    rescale::AbstractConstraint
end

function (U::ScaledNNGradientUpdate{T})(x::T) where T
    n = U.n
    if isfrozen(x, n)
        return x
    end
    F = factors(x)
    Fn = F[n]
    (U.gradientstep)(Fn)
    nnegative!(Fn)
    (U.rescale)(x, n)
end

#=
struct BlockedUpdate{T} <: AbstractUpdate{T}
    updates::NTuple{N, AbstractUpdate}
end
=#

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
