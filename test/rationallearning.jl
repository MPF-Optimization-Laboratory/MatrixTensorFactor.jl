using BlockTensorFactorization
using Random
using LinearAlgebra

using Base: getindex, size
using BlockTensorFactorization: array

######## SymPermTensor ###########

struct SymPermTensor{T, N} <: AbstractArray{T,N}
    nonzeroentry::T # TODO move the argument checking to inside the struct
end

"""
    SymPermTensor(N::Int)

Symmetric permutation tensor or order `n`.

Entries T[i1, ..., iN] are 1/N! if (i1, ..., iN) are a permutation of (1, ..., N), and zero otherwise.
"""
function SymPermTensor(N)
    isinteger(N) || throw(ArgumentError("n must be an integer, got $N"))
    N ≥ 2 || throw(ArgumentError("n must be ≥ 2, got $N"))
    nonzeroentry = 1/factorial(N)
    return SymPermTensor{typeof(nonzeroentry), N}(nonzeroentry)
end

Base.size(_::SymPermTensor{T, N}) where {T, N} = Tuple(fill(N, N)) # 2×2, 3×3×3, 4×4×4×4 etc.
Base.getindex(S::SymPermTensor, i::Int) = S[CartesianIndices(S)[i]]
Base.getindex(S::SymPermTensor, I::CartesianIndex) = getindex(S, Tuple(I)...)
function Base.getindex(S::SymPermTensor{T, N}, I::Vararg{Int}) where {T, N}
    if allunique(I)
        return S.nonzeroentry
    else
        return zero(T)
    end
end

function BlockTensorFactorization.array(S::SymPermTensor{T, N}) where {T, N}
    nonzeroentry = S.nonzeroentry
    output = zeros(T, size(S))
    unique_indices = filter(I -> allunique(Tuple(I)),CartesianIndices(S))
    output[unique_indices] .= nonzeroentry
    return output
end

# @testset "SymPermTensor" begin
#     S2 = SymPermTensor(2)
#     P = [0.0 0.5; 0.5 0.0]
#     @test S2[1] == S2[1,1] == 0
#     @test S2[2] == S2[1,2] == 1/2
#     @test S2[3] == S2[2,1] == 1/2
#     @test S2[4] == S2[2,2] == 0
#     @test S2 == P

#     S3 = SymPermTensor(3)
#     P = [0.0 0.0 0.0; 0.0 0.0 0.16666666666666666; 0.0 0.16666666666666666 0.0;;; 0.0 0.0 0.16666666666666666; 0.0 0.0 0.0; 0.16666666666666666 0.0 0.0;;; 0.0 0.16666666666666666 0.0; 0.16666666666666666 0.0 0.0; 0.0 0.0 0.0]

#     @test S3 == P
# end

########### RationalDecomp ###############

struct RationalDecomp{T, N} <: AbstractDecomposition{T, N}
    factors::Tuple{Vararg{AbstractMatrix{T}}}
    # TODO add size of factor type checking
end

RationalDecomp(factors::Tuple{Vararg{AbstractMatrix{T}}}) where T = RationalDecomp{T, length(factors)}(factors)
frozen(T::RationalDecomp) = false_tuple(nfactors(T))
factors(T::RationalDecomp) = T.factors
core(_::RationalDecomp) = SymPermTensor(2)
rankof(T::RationalDecomposition) = tuple(push!(fill(2, nfactors(T)), ndims(factor(T, 1)))) #TODO fix this last entry, it should be (2,2,2,K) where K is the first layer size

W1 = randn(2,2)
W2 = randn(1,2)
RD = RationalDecomp((W1, W2));

# Example with a (2,2,1) network
function BlockTensorFactorization.array(RD::RationalDecomp)
    coreRD = core(RD)
    top = tuckerproduct(coreRD, Tuple(fill(factor(RD, 1), ndims(RD))))
    bottom = tuckerproduct(coreRD, reverse(factors(RD)))
    return cat(top, bottom; dims=1)
end

function Base.show(io::IO, X::RationalDecomp)
    print(io, typeof(X), "(", factors(X), ")")
    return
end

function Base.show(io::IO, mime::MIME"text/plain", X::RationalDecomp)
    summary(io, X); print(io, " of rank ", rankof(X))
    for (n, f) in zip(eachfactorindex(X), factors(X))
        println(io, "\nFactor ", n, ":")
        show(io, mime, f); flush(io)
    end
end
