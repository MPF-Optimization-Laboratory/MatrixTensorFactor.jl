module BlockTensorDecomposition

# Dependencies
using Random: randn, rand, seed!, shuffle
using LinearAlgebra: ⋅, opnorm, Symmetric, mul!
using DataFrames: DataFrame, nrow

# Method Extentions
using Base: copy, deepcopy, eltype, firstindex, getindex, getproperty, iterate, keys, lastindex, length, show, size, ndims
using Base: +, -, *, /, \ # AbstractDecomposition methods
using Base: ∘, convert # AbstractConstraint methods
using LinearAlgebra: LinearAlgebra, diag

# Basic functionality
include("./utils.jl")
export SuperDiagonal, abs_randn, getnotindex, geomean, identityslice, interlace, norm2, proj_one_hot, projsplx, proj_one_hot!, projsplx!
include("./tensorproducts.jl")
export ×₁, nmp, nmode_product, mtt, slicewise_dot, tuckerproduct, cpproduct

# Low level types and interface
include("./decomposition.jl")
export array, contractions, core, eachfactorindex, factor, factors, frozen, isfrozen, matrix_factor, matrix_factors, rankof
export AbstractDecomposition
export GenericDecomposition, SingletonDecomposition
export AbstractTucker, Tucker, Tucker1, CPDecomposition

include("./objective.jl")
export AbstractObjective
export L2

include("./constraint.jl")
export AbstractConstraint
export check

export GenericConstraint
export ComposedConstraint

export ProjectedNormalization
export l2normalize!, l2normalize_rows!, l2normalize_cols!, l2normalize_1slices!, l2normalize_12slices!
export l1normalize!, l1normalize_rows!, l1normalize_cols!, l1normalize_1slices!, l1normalize_12slices!
export linftynormalize!, linftynormalize_rows!, linftynormalize_cols!, linftynormalize_1slices!, l1normalize_12slices!
export simplex!, simplex_rows!, simplex_cols!, simplex_1slices!, simplex_12slices!

export ScaledNormalization
export l2scaled!, l2scaled_rows!, l2scaled_cols!, l2scaled_1slices!, l2scaled_12slices!
export l1scaled!, l1scaled_rows!, l1scaled_cols!, l1scaled_1slices!, l1scaled_12slices!
export linftynormalize!, linftynormalize_rows!, linftynormalize_cols!, linftynormalize_1slices!, linftynormalize_12slices!
export l1scaled_average12slices!

export EntryWise, IntervalConstraint
export nnegative!, binary!, binaryproject
export l1norm, l2norm, linftynorm

include("./stats.jl")
export AbstractStat
export DisplayDecomposition, EuclidianLipshitz, EuclidianStepSize, FactorNorms, GradientNorm, GradientNNCone
export IterateNormDiff, IterateRelativeDiff, Iteration, ObjectiveValue, ObjectiveRatio, PrintStats, RelativeError

include("./blockupdates.jl")
export AbstractStep
export LipshitzStep, ConstantStep, SPGStep

export AbstractUpdate
export GradientDescent, MomentumUpdate

export ConstraintUpdate
export Projection, NNProjection, SafeNNProjection, Rescale

export BlockedUpdate
export smart_insert!, smart_interlase!

#export block_gradient_decent, nn_block_gradient_decent, scaled_nn_block_gradient_decent, proj_nn_block_gradient_decent

# High level / user-interface
include("./factorize.jl")
export factorize

end # module BlockTensorDecomposition
