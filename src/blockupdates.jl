"""
Mid level code that combines constraints with block updates to be used on an AbstractDecomposition
"""

abstract type AbstractUpdate end
#=
struct Update <: AbstractUpdate
    nothing
end

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
