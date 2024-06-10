module BlockTensorDecomposition

# Dependencies
using Random: randn, rand, seed!
using LinearAlgebra: ⋅, opnorm, Symmetric

# Method Extentions
using Base: size, getindex, show, ndims, *
using LinearAlgebra: LinearAlgebra, diag

# Basic functionality
include("./utils.jl")
export SuperDiagonal
include("./tensorproducts.jl")

# Low level types and interface
include("./decomposition.jl")
export array, contractions, factors, rankof
export AbstractDecomposition, Tucker, Tucker1, CPDecomposition
include("./constraint.jl")
include("./blockupdates.jl")
export least_square_updates

# High level / user-interface
include("./factorize.jl")

end # module BlockTensorDecomposition
