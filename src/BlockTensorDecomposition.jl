module BlockTensorDecomposition

# Dependencies
using Random: randn, rand, seed!, shuffle
using LinearAlgebra: ⋅, opnorm, Symmetric
using DataFrames: DataFrame, nrow

# Method Extentions
using Base: copy, deepcopy, eltype, getindex, iterate, length, show, size, ndims, *
using LinearAlgebra: LinearAlgebra, diag

# Basic functionality
include("./utils.jl")
export SuperDiagonal, abs_randn, identityslice, interlace, norm2
include("./tensorproducts.jl")

# Low level types and interface
include("./decomposition.jl")
export array, contractions, core, eachfactorindex, factor, factors, frozen, isfrozen, matrix_factor, matrix_factors, rankof
export AbstractDecomposition, GenericDecomposition, AbstractTucker, Tucker, Tucker1, CPDecomposition

include("./objective.jl")
export AbstractObjective
export L2

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
export l1scaled_average12slices!

export EntryWise
export nnegative!
export l1norm, l2norm, linftynorm

include("./stats.jl")
export AbstractStat
export GradientNorm, GradientNNCone, ObjectiveValue, ObjectiveRatio, IterateNormDiff, IterateRelativeDiff, Iteration

include("./blockupdates.jl")
export AbstractStep
export LipshitzStep, ConstantStep, SPGStep

export AbstractUpdate
export GradientDescent, MomentumUpdate, Rescale, Projection
export BlockedUpdate
export smart_insert!, smart_interlase!

#export block_gradient_decent, nn_block_gradient_decent, scaled_nn_block_gradient_decent, proj_nn_block_gradient_decent

# High level / user-interface
include("./factorize.jl")
export factorize

end # module BlockTensorDecomposition
