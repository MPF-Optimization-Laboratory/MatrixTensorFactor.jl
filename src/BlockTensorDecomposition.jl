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
export array, contractions, core, factors, frozen, isfrozen, matrix_factors, rankof
export AbstractDecomposition, GenericDecomposition, Tucker, Tucker1, CPDecomposition
include("./constraint.jl")
include("./blockupdates.jl")
export AbstractUpdate

# High level / user-interface
include("./factorize.jl")

end # module BlockTensorDecomposition
