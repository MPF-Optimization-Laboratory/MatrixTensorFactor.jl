module BlockTensorDecomposition

# Dependencies
using Random: randn, rand, seed!, shuffle
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
export array, contractions, core, factor, factors, frozen, isfrozen, matrix_factors, rankof
export AbstractDecomposition, GenericDecomposition, Tucker, Tucker1, CPDecomposition

include("./constraint.jl")
export AbstractConstraint
export check

export GenericConstraint

export ProjectedNormalization
export l2normalize!, l2normalize_rows!, l2normalize_cols!, l2normalize_1slices!, l2normalize_12slices!
export l1normalize!, l1normalize_rows!, l1normalize_cols!, l1normalize_1slices!, l1normalize_12slices!
export linftynormalize!, linftynormalize_rows!, linftynormalize_cols!, linftynormalize_1slices!, l1normalize_12slices!

export ScaledNormalization
export l2scaled!, l2scaled_rows!, l2scaled_cols!, l2scaled_1slices!, l2scaled_12slices!
export l1scaled!, l1scaled_rows!, l1scaled_cols!, l1scaled_1slices!, l1scaled_12slices!
export linftynormalize!, linftynormalize_rows!, linftynormalize_cols!, linftynormalize_1slices!, linftynormalize_12slices!

export EntryWise
export nnegative!

include("./blockupdates.jl")
export AbstractUpdate

export block_gradient_decent

# High level / user-interface
include("./factorize.jl")

end # module BlockTensorDecomposition
