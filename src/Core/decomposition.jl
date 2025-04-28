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
Base.getindex(D::AbstractDecomposition, i::Int) = D[CartesianIndices(D)[i]] # usually more efficient to compute a single entry given the cartesian index rather than linear index
Base.getindex(D::AbstractDecomposition, I::CartesianIndex) = getindex(D, Tuple(I)...)
Base.getindex(D::AbstractDecomposition, I::Vararg{Int}) = getindex(array(D), I...)

# copy/deepcopy all factors and any other properties before reconstructing
Base.copy(D::AbstractDecomposition) = typeof(D)((copy.(getfield(D, p)) for p in propertynames(D))...)
Base.deepcopy(D::AbstractDecomposition) = typeof(D)((deepcopy.(getfield(D, p)) for p in propertynames(D))...)

#Mathematical Operations
Base.:+(A::AbstractDecomposition, B::AbstractArray) = array(A) + B
Base.:+(A::AbstractArray, B::AbstractDecomposition) = A + array(B)
Base.:+(A::AbstractDecomposition, B::AbstractDecomposition) = array(A) + array(B)

Base.:-(A::AbstractDecomposition, B::AbstractArray) = array(A) - B
Base.:-(A::AbstractArray, B::AbstractDecomposition) = A - array(B)
Base.:-(A::AbstractDecomposition, B::AbstractDecomposition) = array(A) - array(B)
Base.:-(A::AbstractDecomposition) = -array(A)

Base.:*(A::AbstractDecomposition, B::AbstractArray) = array(A) * B
Base.:*(A::AbstractArray, B::AbstractDecomposition) = A * array(B)
Base.:*(A::AbstractDecomposition, B::AbstractDecomposition) = array(A) * array(B)

Base.:/(A::AbstractDecomposition, B::AbstractArray) = array(A) / B
Base.:/(A::AbstractArray, B::AbstractDecomposition) = A / array(B)
Base.:/(A::AbstractDecomposition, B::AbstractDecomposition) = array(A) / array(B)

Base.:\(A::AbstractDecomposition, B::AbstractArray) = array(A) \ B
Base.:\(A::AbstractArray, B::AbstractDecomposition) = A \ array(B)
Base.:\(A::AbstractDecomposition, B::AbstractDecomposition) = array(A) \ array(B)

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
eachfactorindex(D::AbstractDecomposition) = 1:nfactors(D)

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

"""
SingletonDecomposition(A::AbstractArray, frozen=false)

Wraps an AbstractArray so it can be treated like an AbstractDecomposotition
"""
struct SingletonDecomposition{T, N} <: AbstractDecomposition{T, N}
    factors::Tuple{AbstractArray{T}}
    frozen::Tuple{Bool}
end

SingletonDecomposition(A::AbstractArray, frozen=false) = SingletonDecomposition{eltype(A),ndims(A)}((A,), (frozen,))
contractions(_::SingletonDecomposition) = ()
array(S::SingletonDecomposition) = factors(S)[begin]

#Base.show(io::IO, D::AbstractDecomposition) = show.((io,), factors(D))

# Tucker decompositions
abstract type AbstractTucker{T, N} <: AbstractDecomposition{T, N} end

function Base.show(io::IO, X::AbstractTucker)
    print(io, typeof(X), "(", factors(X), ",", frozen(X), ")")
    return
end

function Base.show(io::IO, mime::MIME"text/plain", X::AbstractTucker)
    summary(io, X); print(io, " of rank ", rankof(X))
    for (n, f) in zip(eachfactorindex(X), factors(X))
        println(io, "\nFactor ", n, ":")
        show(io, mime, f); flush(io)
    end
end

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
        core = factors[begin]
        ndims(core) == N ||
            throw(ArgumentError("Core should have matching number of dims as the call to Tucker1{$T, $N}, got $(ndims(core))"))

        _valid_tucker(factors) ||
            throw(ArgumentError("Not a valid Tucker decomposition"))

        length(frozen) == length(factors) ||
            throw(ArgumentError("Tuple of frozen factors length $(length(frozen)) does not match number of factors $(length(factors))"))

        new{T, N}(factors, frozen)
    end
end

function _valid_tucker(factors)
    # Need one factor for each core dimention
    core, other_factors... = factors
    if ndims(core) != length(other_factors)
        @warn "Core is order $(ndims(factors[1])) but got $(length(factors)-1) other factor(s)"
        return false
    end

    # Need the core sizes to match the second dimention of each other factor
    core_size = size(core)
    other_sizes = map(x -> size(x, 2), other_factors)
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
        core = factors[begin]
        ndims(core) == N ||
            throw(ArgumentError("Core should have matching number of dims as the call to Tucker1{$T, $N}, got $(ndims(core))"))

        core_dim1 = size(core, 1)
        matrix_dim2 = size(factors[end], 2)
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
function Tucker(full_size::NTuple{N, Integer}, ranks::NTuple{N, Integer}; frozen=false_tuple(length(ranks)+1), init=DEFAULT_INIT, kwargs...) where N
    core = init(ranks)
    matrix_factors = init.(full_size, ranks)
    Tucker((core, matrix_factors...), frozen)
end

# TODO throw a readable error if the length of `ranks` does not match the number of dimensions of full_size

function Tucker1(full_size::NTuple{N, Integer}, rank::Integer; frozen=false_tuple(2), init=DEFAULT_INIT, kwargs...) where N
    I, J... = full_size
    core = init((rank, J...))
    matrix_factor = init(I, rank)
    Tucker1((core, matrix_factor), frozen)
end


# AbstractTucker interface
core(T::AbstractTucker) = factors(T)[begin]
matrix_factors(T::AbstractTucker) = factors(T)[begin+1:end]
matrix_factor(T::AbstractTucker, n::Integer) = matrix_factors(T)[n]
isfrozen(T::AbstractTucker, n::Integer) = frozen(T)[n+1]
# This way, the 1st matrix factor in a CPDecomposition is factors(T)[1]
# and the 1st matrix factor in a Tucker is factors(T)[2]

"""
    eachrank1term(T::AbstractTucker)

Creates a generator for each rank 1 term of a Tucker decomposition.
"""
eachrank1term(T::AbstractTucker) = error("eachrank1term is not yet implemented for AbstractTuckers of type $(typeof(T))")

"""
    eachrank1term(T::Tucker1)

The (Tucker-1) rank-1 tensors Tr[i1, ..., iN] = A[i1,r] * B[r, i2, ..., iN] for each r = 1, ..., rankof(T).
"""
eachrank1term(T::Tucker1) = (Ar .* reshape_ndims(Br, ndims(T)) for (Br, Ar) in zip(eachslice(core(T); dims=1), eachcol(matrix_factor(T, 1))))

# AbstractDecomposition Interface
array(T::AbstractTucker) = multifoldl(contractions(T), factors(T))
array(T::Tucker) = tuckerproduct(core(T), matrix_factors(T))
factors(T::AbstractTucker) = T.factors
contractions(T::Tucker) = tucker_contractions(ndims(T))
contractions(_::Tucker1) = ((×₁),)
rankof(T::Tucker) = map(x -> size(x, 2), matrix_factors(T))
rankof(T::Tucker1) = size(core(T), 1)

# Essentialy zero index tucker factors so the core is the 0th factor, and the nth factor
# is the matrix factor in the nth dimention
function factor(D::AbstractTucker, n::Integer)
    if n == 0
        return core(D)
    elseif n >= 1
        return matrix_factor(D, n)
    else
        throw(ArgumentError("No $(n)th factor in $(typeof(D))"))
    end
end

eachfactorindex(D::AbstractTucker) = 0:(nfactors(D)-1) # 0 based, where core is 0

# AbstractArray interface
# Efficient size and indexing for CPDecomposition
Base.size(T::Tucker) = map(x -> size(x, 1), matrix_factors(T))
Base.size(T::Tucker1) = (size(matrix_factors(T)[begin], 1), size(core(T))[begin+1:end]...)

Base.getindex(T::Tucker, i::Int) = T[CartesianIndices(T)[i]]
Base.getindex(T::Tucker, I::Vararg{Int}) = _gettuckerindex(T, I)
_gettuckerindex(T::Tucker, I) = _gettuckerindex(core(T), matrix_factors(T), I)

Base.getindex(T::Tucker1, i::Int) = T[CartesianIndices(T)[i]]
function Base.getindex(T::Tucker1, I::Vararg{Int})
    G, A = factors(T)
    i, J... = I # (i, J) = (I[1], I[begin+1:end])
    return (@view A[i, :]) ⋅ view(G, :, J...)
end

"""
CP decomposition. Takes the form of an outerproduct of multiple matrices.

For example, a rank r CP decomposition of an order three tensor D would be, entry-wise,
D[i, j, k] = sum_r A[i, r] * B[j, r] * C[k, r]).

CPDecomposition((A, B, C))
"""
struct CPDecomposition{T, N} <: AbstractTucker{T, N}
	factors::NTuple{N, <:AbstractMatrix{T}} # ex. (A, B, C)
    frozen::NTuple{N, Bool}
    function CPDecomposition{T, N}(factors, frozen) where {T, N}
        ranks = map(x -> size(x, 2), factors)
        allequal(ranks) ||
            throw(ArgumentError("Second dimention of factors should be equal. Got $ranks"))

        length(frozen) == length(factors) ||
            throw(ArgumentError("Tuple of frozen factors length $(length(frozen)) does not match number of factors $(length(factors))"))

        new{T, N}(factors, frozen)
    end
end

# Constructor
CPDecomposition(factors, frozen=false_tuple(length(factors))) = CPDecomposition{eltype(factors[begin]), length(factors)}(factors, frozen)
function CPDecomposition(full_size::Tuple{Vararg{Integer}}, rank::Integer; frozen=false_tuple(length(full_size)), init=DEFAULT_INIT, kwargs...)
    factors = init.(full_size, rank)
    CPDecomposition(factors, frozen)
end

# AbstractDecomposition Interface
factors(CPD::CPDecomposition) = CPD.factors
array(CPD::CPDecomposition) = cpproduct(factors(CPD))
frozen(CPD::CPDecomposition) = CPD.frozen
vector_outer(v) = reshape(kron(reverse(v)...),length.(v))
eachfactorindex(CPD::CPDecomposition) = 1:nfactors(CPD) # unlike other AbstractTucker's, back to 1 based since there's only matrix factors
isfrozen(CPD::CPDecomposition, n::Integer) = n == 0 ? true : frozen(CPD)[n] # similar to eachfactorindex

# AbstractTucker Interface
matrix_factors(CPD::CPDecomposition) = factors(CPD)
core(CPD::CPDecomposition{T, N}) where {T, N} = identity_tensor(T, rankof(CPD), N) #SuperDiagonal(ones(eltype(CPD), rankof(CPD)), ndims(CPD))

"""
    eachrank1term(T::CPDecomposition)

The (CP) rank-1 tensors Tr[i1, ..., iN] = A1[i1, r] * ... * AN[iN, r]  for each r = 1, ..., rankof(T).
"""
eachrank1term(CPD::CPDecomposition) = (reduce(.*, reshape_ndims(col, i) for (i, col) in enumerate(cols)) for cols in zip((map(eachcol, factors(CPD)))...))

# Efficient size and indexing for CPDecomposition
Base.size(CPD::CPDecomposition) = map(x -> size(x, 1), factors(CPD))
# Example: CPD[i, j, k] = sum(A[i, :] .* B[j, :] .* C[k, :])
Base.getindex(CPD::CPDecomposition, i::Int) = CPD[CartesianIndices(CPD)[i]]
Base.getindex(CPD::CPDecomposition, I::Vararg{Int}) = sum(reduce(.*, (@view f[i,:]) for (f,i) in zip(factors(CPD), I)))

# Additional CPDecomposition interface
"""The single rank for a CP Decomposition"""
rankof(CPD::CPDecomposition) = size(matrix_factors(CPD)[begin], 2)

# TODO Define a matrix factorization type Y=AB so that thinking about tensors can be avoided
# Could even exploit Tucker1 or CPDecomposition to do this
