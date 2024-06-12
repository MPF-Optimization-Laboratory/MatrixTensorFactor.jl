"""
Low level code defining the verious decomposition types like Tucker and CP
"""

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
    factor(D::AbstractDecomposition, n::Integer)

A tuple of (usually smaller) arrays representing the decomposition of a (usually larger)
array. Use `factor(D, n)` to get just the `n`'th factor.
"""
factors(D::AbstractDecomposition) = D.factors
factor(D::AbstractDecomposition, n::Integer) = factors(D)[n]

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
    frozen(D::AbstractDecomposition)

A tuple of `Bools` the same length as `factors(D)` showing which factors are "frozen" in the
sense that a block decent algorithm should skip these factors when decomposing a tensor.
"""
frozen(D::AbstractDecomposition) = D.frozen

isfrozen(D::AbstractDecomposition) = any(frozen(D))
isfrozen(D::AbstractDecomposition, n::Integer) = frozen(D)[n]

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

DEFAULT_INIT = randn

"""
Most general decomposition. Takes the form of interweaving contractions between the factors.

For example, T = A * B + C could be represented as GenericDecomposition((A, B, C), (*, +))
"""
struct GenericDecomposition{T, N} <: AbstractDecomposition{T, N}
	factors::Tuple{Vararg{AbstractArray{T}}} # ex. (A, B, C)
	contractions::Tuple{Vararg{Function}}
    frozen::Tuple{Vararg{Bool}}
end

# AbstractDecomposition Interface
array(G::GenericDecomposition) = multifoldl(contractions(G), factors(G))
factors(G::GenericDecomposition) = G.factors
contractions(G::GenericDecomposition) = G.contractions
frozen(G::GenericDecomposition) = G.frozen

function multifoldl(ops, args)
    @assert (length(ops) + 1) == length(args)
    x = args[begin]
    for (op, arg) in zip(ops, args[begin+1:end]) # TODO want @view args[begin+1:end] when possible
        x = op(x, arg)
    end
    return x
end

#Base.show(io::IO, D::AbstractDecomposition) = show.((io,), factors(D))

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
    frozen::NTuple{M, Bool} where M
    function Tucker{T, N}(factors, frozen) where {T, N}
        _valid_tucker(factors) ||
            throw(ArgumentError("Not a valid Tucker decomposition"))

        length(frozen) == length(factors) ||
            throw(ArgumentError("Tuple of frozen factors length $(length(frozen)) does not match number of factors $(length(factors))"))

        new{T, N}(factors)
    end
end

function _valid_tucker(factors)
    # Need one factor for each core dimention
    core = factors[begin]
    other_factors = factors[begin+1:end]
    if ndims(core) != length(other_factors)
        @warn "Core is order $(ndims(factors[1])) but got $(length(factors)-1) other factor(s)"
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
    frozen::Tuple{Bool, Bool}
    function Tucker1{T, N}(factors, frozen) where {T, N}
        core_dim1 = size(factors[begin])[1]
        matrix_dim2 = size(factors[end])[2]
        if core_dim1 != matrix_dim2
            @warn "First core dimention $core_dim1 does not match second matrix dimention $matrix_dim2"
            throw(ArgumentError("Not a valid Tucker1 decomposition"))
        end

        length(frozen) == length(factors) ||
            throw(ArgumentError("Tuple of frozen factors length $(length(frozen)) does not match number of factors $(length(factors))"))

        new{T, N}(factors, frozen)
    end
end

# TODO add automatic struct convertion for Tucker-n beyond Tucker-1 when the number of other
# factors is less than the number of dimentions of the core

# Constructors
Tucker(factors::Tuple{Vararg{AbstractArray{T}}}, frozen=false_tuple(length(factors))) where T = Tucker{T, length(factors) - 1}(factors, frozen)
#Tucker(factors::Tuple{<:AbstractArray{T}, <:AbstractMatrix{T}}, frozen=false_tuple(2)) where T = Tucker1(factors, frozen) # use the more specific struct
Tucker1(factors::Tuple{<:AbstractArray{T}, <:AbstractMatrix{T}}, frozen=false_tuple(2)) where T = Tucker1{T, ndims(factors[1])}(factors, frozen)
function Tucker(full_size::NTuple{N, Integer}, ranks::NTuple{N, Integer}; frozen=false_tuple(length(ranks)+1), init=DEFAULT_INIT) where N
    core = init(ranks)
    matrix_factors = init.(full_size, ranks)
    Tucker((core, matrix_factors...), frozen)
end

function Tucker1(full_size::NTuple{N, Integer}, rank::Integer; frozen=false_tuple(2), init=DEFAULT_INIT) where N
    I, J... = full_size
    core = init((rank, J...))
    matrix_factor = init(I, rank)
    Tucker1((core, matrix_factor), frozen)
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
CP decomposition. Takes the form of an outerproduct of multiple matricies.

For example, a rank r CP decomposition of an order three tensor D would be, entry-wise,
D[i, j, k] = sum_r A[i, r] * B[j, r] * C[k, r]).

CPDecomposition((A, B, C))
"""
struct CPDecomposition{T, N} <: AbstractTucker{T, N}
	factors::NTuple{N, <:AbstractMatrix{T}} # ex. (A, B, C)
    frozen::NTuple{N, Bool}
    function CPDecomposition{T, N}(factors, frozen) where {T, N}
        ranks = map(x -> size(x)[2], factors)
        allequal(ranks) ||
            throw(ArgumentError("Second dimention of factors should be equal. Got $ranks"))

        length(frozen) == length(factors) ||
            throw(ArgumentError("Tuple of frozen factors length $(length(frozen)) does not match number of factors $(length(factors))"))

        new{T, N}(factors, frozen)
    end
end

# Constructor
CPDecomposition(factors, frozen=false_tuple(length(factors))) = CPDecomposition{eltype(factors[begin]), length(factors)}(factors, frozen)
function CPDecomposition(full_size::Tuple{Vararg{Integer}}, rank::Integer; frozen=false_tuple(length(full_size)), init=DEFAULT_INIT)
    factors = init.(full_size, rank)
    CPDecomposition(factors, frozen)
end

# AbstractDecomposition Interface
factors(CPD::CPDecomposition) = CPD.factors
array(CPD::CPDecomposition) = mapreduce(vector_outer, +, zip((eachcol.(factors(CPD)))...))
frozen(CPD::CPDecomposition) = CPD.frozen
vector_outer(v) = reshape(kron(reverse(v)...),length.(v))

# AbstractTucker Interface
matrix_factors(CPD::CPDecomposition) = factors(CPD)
core(CPD::CPDecomposition) = SuperDiagonal(ones(eltype(CPD), rankof(CPD)), ndims(CPD))

# Efficient size and indexing for CPDecomposition
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
