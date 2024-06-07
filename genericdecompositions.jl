# Method extentions
using Base: size, getindex, show, ndims, *

using Random
using LinearAlgebra: ⋅, opnorm, Symmetric

"""
Abstract type for the different decompositions.

Main interface for subtypes are the following functions.
Required:
    `array(D)`: how to construct the full array from the decomposition representation
    `factors(D)`: a tuple of arrays, the decomposed factors
Optional:
    `getindex(D, i::Int)` and `getindex(D, I::Vararg{Int, N})`: how to get the ith or Ith
        element of the reconstructed array. Defaults to `getindex(array(D), x)`, but there
        is often a more efficient way to get a specific element from large tensors.
    `size(D)`: Defaults to `size(array(D))`.
"""
abstract type AbstractDecomposition{T, N} <: AbstractArray{T, N} end

# Fallback interface for any AbstractDecomposition to behave like an AbstractArray
Base.size(D::AbstractDecomposition) = size(array(D))
Base.ndims(_::AbstractDecomposition{T, N}) where {T, N} = N
Base.getindex(D::AbstractDecomposition, i::Int) = getindex(array(D), i)
Base.getindex(D::AbstractDecomposition, I::Vararg{Int}) = getindex(array(D), I...)

# AbstractDecomposition functions
"""
    nfactors(D::AbstractDecomposition)

Returns the number of factors/blocks in a decomposition.
"""
nfactors(D::AbstractDecomposition) = length(factors(D))

"""
    array(D::AbstractDecomposition)

Turns a decomposition into the full array, usually by multiplying the factors to reconstruct
the full array.
"""
array(D::AbstractDecomposition) = Array(D)::AbstractArray


"""
    factors(D::AbstractDecomposition)

A tuple of (usually smaller) arrays representing the decomposition of a (usually larger)
array.
"""
factors(D::AbstractDecomposition) = D.factors

"""
    contractions(D::AbstractDecomposition)

A tuple of functions defining a recipe for reconstructing a full array from the factors
of the decomposition.

Example
-------
(op1, op2) = contractions(D)
(A, B, C) = factors(D)

array(D) == (A op1 B) op2 C
"""
contractions(D::AbstractDecomposition) = D.contractions

"""
    rankof(D::AbstractDecomposition)

Internal dimention sizes for a decomposition. Returns the sizes of all factors if not defined
for a concrete subtype of `AbstractDecomposition`.

Examples
--------
`CPDecomposition`: size of second dimention for the factors
`Tucker`: size of the core factor
`Tucker1`: size of the first dimention of the core factor
"""
rankof(D::AbstractDecomposition) = size.(factors(D))

DEFAULT_INIT(x...) = randn(x...)

"""
Most general decomposition. Takes the form of interweaving contractions between the factors.

For example, T = A * B + C could be represented as GenericDecomposition((A, B, C), (*, +))
"""
struct GenericDecomposition{T, N} <: AbstractDecomposition{T, N}
	factors::Tuple{Vararg{<:AbstractArray{T}}} # ex. (A, B, C)
	contractions::Tuple{Vararg{<:Function}}
end

# AbstractDecomposition Interface
array(G::GenericDecomposition) = multifoldl(contractions(G), factors(G))
factors(G::GenericDecomposition) = G.factors
contractions(G::GenericDecomposition) = G.contractions

function multifoldl(ops, args)
    @assert (length(ops) + 1) == length(args)
    x = args[begin]
    for (op, arg) in zip(ops, args[begin+1:end]) # TODO want @view args[begin+1:end] when possible
        x = op(x, arg)
    end
    return x
end

#Base.show(io::IO, D::AbstractDecomposition) = show.((io,), factors(D))

"""
CP decomposition. Takes the form of an outerproduct of multiple matricies.

For example, a rank r CP decomposition of an order three tensor D would be, entry-wise,
D[i, j, k] = sum_r A[i, r] * B[j, r] * C[k, r]).

CPDecomposition((A, B, C))
"""
struct CPDecomposition{T, N} <: AbstractDecomposition{T, N}
	factors::NTuple{N, <:AbstractArray{T}} # ex. (A, B, C)
    # TODO Constrain size(factors(CPD)[i])[2] for all i to be equal
end

# Constructor
CPDecomposition(factors) = CPDecomposition{eltype(factors[begin])}(factors)
function CPDecomposition(full_size::Tuple{Vararg{Integer}}, rank::Integer; init=DEFAULT_INIT)
    factors = init.(full_size, rank)
    CPDecomposition(factors)
end

# AbstractDecomposition Interface
factors(CPD::CPDecomposition) = CPD.factors
array(CPD::CPDecomposition) = mapreduce(vector_outer, +, zip((eachcol.(factors(CPD)))...))
vector_outer(v) = reshape(kron(reverse(v)...),length.(v))

# Efficient size and indexing for CPDecomposition
# Base.ndims(CPD::CPDecomposition) = length(factors(CPD))
Base.size(CPD::CPDecomposition) = map(x -> size(x)[1], factors(CPD))
# Example: CPD[i, j, k] = sum(A[i, :] .* B[j, :] .* C[k, :])
Base.getindex(CPD::CPDecomposition, I::Vararg{Int})= sum(reduce(.*, (@view f[i,:]) for (f,i) in zip(factors(CPD), I)))

# Additional CPDecomposition interface
"""The single rank for a CP Decomposition"""
rankof(CPD::CPDecomposition) = size(factors(CPD)[begin])[2]

function Base.show(io::IO, CPD::CPDecomposition)
    println(io, size(CPD), " rank ", rankof(CPD), " ", typeof(CPD), ":")
    display.(io, F)
    return
end

# Tucker decompositions
abstract type AbstractTucker{T, N} <: AbstractDecomposition{T, N} end

"""
Tucker decomposition. Takes the form of a core times a matrix for each dimention.

For example, a rank (r, s, t) Tucker decomposition of an order three tensor D would be, entry-wise,
D[i, j, k] = sum_r sum_s sum_t G[r, s, t] * A[i, r] * B[j, s] * C[k, t]).

CPDecomposition((A, B, C))
"""
struct Tucker{T, N} <: AbstractTucker{T, N}
	factors::Tuple{AbstractArray{T}, Vararg{AbstractMatrix{T}}} # ex. (G, A, B, C)
    Tucker{T, N}(factors) where {T, N} = !_valid_tucker(factors) ? throw(ArgumentError("Not a valid Tucker decomposition")) : new{T, N}(factors)
end

function _valid_tucker(factors)
    # Need one factor for each core dimention
    core = factors[begin]
    other_factors = factors[begin+1:end]
    if ndims(core) != length(other_factors)
        @warn "Core is order $(ndims(factors[1])) but got $(length(factors)-1) other factors"
        return false
    end

    # Need the core sizes to match the second dimention of each other factor
    core_size = size(core)
    other_sizes = map(x -> size(x)[2], other_factors)
    if any(core_size .!= other_sizes)
        @warn "Size of core $(size(core)) is not compatible with the other factor's dimentions $other_sizes"
        return false
    end

    return true
end

struct Tucker1{T, N} <: AbstractTucker{T, N}
	factors::Tuple{<:AbstractArray{T}, <:AbstractMatrix{T}} # ex. (G, A)
    function Tucker1{T, N}(factors) where {T, N}
        core_dim1 = size(factors[begin])[1]
        matrix_dim2 = size(factors[end])[2]
        if core_dim1 != matrix_dim2
            @warn "First core dimention $core_dim1 does not match second matrix dimention $matrix_dim2"
            throw(ArgumentError("Not a valid Tucker1 decomposition"))
        end
        new{T, N}(factors)
    end
end

# TODO add automatic struct convertion for Tucker-n beyond Tucker-1 when the number of other
# factors is less than the number of dimentions of the core

# Constructors
Tucker(factors::Tuple{Vararg{<:AbstractArray{T}}}) where T = Tucker{T, length(factors) - 1}(factors)
Tucker(factors::Tuple{<:AbstractArray{T}, <:AbstractMatrix{T}}) where T = Tucker1(factors) # use the more specific struct
Tucker1(factors::Tuple{<:AbstractArray{T}, <:AbstractMatrix{T}}) where T = Tucker1{T, ndims(factors[1])}(factors)
function Tucker(full_size::NTuple{N, Integer}, ranks::NTuple{N, Integer}; init=DEFAULT_INIT) where N
    core = init(ranks)
    matrix_factors = init.(full_size, ranks)
    Tucker((core, matrix_factors...))
end

function Tucker1(full_size::NTuple{N, Integer}, rank::Integer; init=DEFAULT_INIT) where N
    I, J... = full_size
    core = init((rank, J...))
    matrix_factor = init(I, rank)
    Tucker((core, matrix_factor))
end

# AbstractTucker interface
core(T::AbstractTucker) = factors(T)[begin]
matrix_factors(T::AbstractTucker) = factors(T)[begin+1:end]

# AbstractDecomposition Interface
array(T::AbstractTucker) = multifoldl(contractions(T), factors(T))
factors(T::AbstractTucker) = T.factors
contractions(T::Tucker) = Tuple((G, A) -> nmp(G, A, n) for n in 1:ndims(T))
contractions(_::Tucker1) = ((×₁),)
rankof(T::Tucker) = map(x -> size(x)[2], matrix_factors(T))
rankof(T::Tucker1) = size(core(T))[begin]

# AbstractArray interface
# Base.ndims(T::AbstractTucker) = ndims(core(T))
# Efficient size and indexing for CPDecomposition
Base.size(T::Tucker) = map(x -> size(x)[1], matrix_factors(T))
Base.size(T::Tucker1) = (size(factors(T)[2])[1], size(core(T))[begin+1:end]...)
function Base.getindex(T::Tucker1, I::Vararg{Int})
    G, A = factors(T)
    i = I[1]
    J = I[begin+1:end]
    return (@view A[i, :]) ⋅ view(G, :, J...)
end
# Example: D[i, j, k] = sum_r sum_s sum_t G[r, s, t] * A[i, r] * B[j, s] * C[k, t])
# which is like the single product
#=
function Base.getindex(T::Tucker, I::Vararg{Int})
    matrix_factors_slice = ((@view f[i,:]) for (f,i) in zip(factors(T)[begin+1:end], I))
    ops = Tuple((×₁) for _ in 1:ndims(T)) # the leading index gets collapsed each time, so it is always the 1 mode product
    return multifoldl(ops, (core(T), matrix_factors_slice...))
end=#

"""
    nmode_product(A::AbstractArray, B::AbstractMatrix, n::Integer)

This is defined for arbitrary n, which cannot be done using Einsum/Tullio since they can
only handle fixed ordered-tensors.
"""
function nmode_product(A::AbstractArray, B::AbstractMatrix, n::Integer)
    Aperm = swapdims(A, n)
    Cperm = Aperm ×₁ B # convert the problem to the mode-1 product
    return swapdims(Cperm, n) # swap back
    #Cmat = B * mat(A, n)
    #sizeA = size(A)
    #sizeC = (sizeA[begin:n-1]..., size(B)[2], sizeA[n+1:end]...)
    #return imat(Cmat, n, sizeC)
end

function nmode_product(A::AbstractArray, b::AbstractVector, n::Integer)
    Aperm = swapdims(A, n)
    Cperm = Aperm ×₁ b # convert the problem to the mode-1 product
    return Cperm # no need to swap since the first dimention is dropped
end

nmp = nmode_product # Short-hand alias

"""
    swapdims(A::AbstractArray, a::Integer, b::Integer=1)

Swap dimentions `a` and `b`.
"""
function swapdims(A::AbstractArray, a::Integer, b::Integer=1)
    dims = collect(1:ndims(A)) # TODO construct the permutation even more efficiently
    dims[a] = b; dims[b] = a
    permutedims(A, dims)
end

"""1-mode product between a tensor and a matrix"""
×₁(A::AbstractArray, B::AbstractMatrix) = mtt(B, A)
×₁(A::AbstractArray, b::AbstractVector) = dropdims(mtt(b', A); dims=1)

"""
Matrix times Tensor

Looks like C[i1, i2, ..., iN] = sum_r A[i1, r] * B[r, i2, ..., iN] entry-wise.
"""
function mtt(A::AbstractMatrix, B::AbstractArray)
    sizeB = size(B)
    Bmat = reshape(B, sizeB[1], :)
    Cmat = A * Bmat
    C = reshape(Cmat, size(A)[1], sizeB[2:end]...)
    return C
end

#####################################################

struct BlockUpdatedDecomposition{T, N} <: AbstractDecomposition{T,N}
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

"""
    slicewise_dot(A::AbstractArray, B::AbstractArray)

Constracts all but the first dimentions of A and B by performing a dot product over each `dim=1` slice.

Generalizes `@einsum C[s,r] := A[s,j,k]*B[r,j,k]` to arbitrary dimentions.
"""
function slicewise_dot(A::AbstractArray, B::AbstractArray)
    C = zeros(size(A)[1], size(B)[1]) # Array{promote_type(T, U), 2}(undef, size(A)[1], size(B)[1]) doesn't seem to be faster

    if A === B # use the faster routine if they are the same array
        return _slicewise_self_dot!(C, A)
    end

    for (i, A_slice) in enumerate(eachslice(A, dims=1))
        for (j, B_slice) in enumerate(eachslice(B, dims=1))
            C[i, j] = A_slice ⋅ B_slice
        end
    end
    return C
end

function _slicewise_self_dot!(C, A)
    enumerated_A_slices = enumerate(eachslice(A, dims=1))
    for (i, Ai_slice) in enumerated_A_slices
        for (j, Aj_slice) in enumerated_A_slices
            if i > j
                continue
            else
                C[i, j] = Ai_slice ⋅ Aj_slice
            end
        end
    end
    return Symmetric(C)
end


#####################################################

abstract type AbstractConstraint <: Function end

"""
    GenericConstraint <: AbstractConstraint

General constraint. Simply applies apply and checks with check. Composing any two
`AbstractConstraint`s will return this type.

Calling a `GenericConstraint` on an `AbstractDecomposition` will use the function in the
field `apply`. Use `check(A, C::GenericConstraint)` to use the function in the field `check`.
"""
struct GenericConstraint <: AbstractConstraint
    apply::Function # input a AbstractDecomposition -> mutate it so that `check` would return true
    check::Function
end

function (C::GenericConstraint)(D::AbstractDecomposition)
    (C.apply)(D)
end

check(A::AbstractDecomposition, C::GenericConstraint) = (C.check)(A)

∘(f::AbstractConstraint, g::AbstractConstraint) = GenericConstraint(f.apply ∘ g.apply, f.check ∘ g.check)

"""

"""
struct Normalization <: AbstractConstraint
    apply::Function # input a AbstractDecomposition -> mutate it so that `check` would return true
    check::Function # input a AbstractDecomposition -> output a Bool
end


"""Entrywise constraint. Note both apply and check needs to be performed entrywise on an array"""
struct EntryWise <: AbstractConstraint
    apply::Function
    check::Function
end

"""Make entrywise callable, by applying the constraint entrywise to arrays"""
function (C::EntryWise)(A::AbstractArray)
    A .= (C.apply).(A)
end

"""
    check(A::AbstractArray, C::EntryWise)::Bool

Checks if `A` is entrywise constrained
"""
check(A::AbstractArray, C::EntryWise) = all((C.check).(A))

const nnegative! = EntryWise(x -> max(0, x), x -> x >= 0)



#####################################################
"""
mat(A::AbstractArray, n::Integer)

Matricize along the nth mode.
"""
function mat(A::AbstractArray, n::Integer)
N = ndims(A)
1 <= n && n <= N || throw(ArgumentError("n=$n is not a valid dimention to matricize"))
dims = ((1:n-1)..., (n+1:N)...) # all axis, skipping n
return cat(eachslice(A; dims)...; dims=2) # TODO Idealy return a view/SubArray
end

"""Khatri-Rao product. A ⊙ B can be typed with `\\odot`."""
⊙(A, B) = khatrirao(A, B)
khatrirao(A, B) = hcat(kron.(eachcol(A), eachcol(B))...)

#fullouter(v...) = reshape(kron(reverse(vec.(v))...),tuple(vcat(collect.(size.(v))...)...))

#######################################################
# Tests

A = randn(3,3);
B = randn(4,3);
C = randn(5,3);

CPD = CPDecomposition((A, B, C))

CPD = CPDecomposition((A, B))

G = randn(3,3,3)

G = Tucker((G, A, B, C))

G = Tucker((G, A))

G = Tucker((B', A))

G = Tucker1((10,11,12), 5);
Y = Tucker1((10,11,12), 5);

BUD = least_square_updates(G, Y)
