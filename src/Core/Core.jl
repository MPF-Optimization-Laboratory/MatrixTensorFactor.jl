module Core

# Dependencies
using Random: randn, rand, seed!, shuffle
using LinearAlgebra: ⋅, opnorm, Symmetric, mul!, Diagonal, I, norm
using DataFrames: DataFrame, nrow

const id = I # often use `I` for a dimension size, e.g. i ∈ 1:I, so here's a way to get the identity matrix

# Method Extensions
using Base: copy, deepcopy, eltype, filter, firstindex, getindex, getproperty, iterate, keys, lastindex, length, show, size, ndims
using Base: +, -, *, /, \ # AbstractDecomposition methods
using Base: ^, √, min # DiagonalTuple method
using Base: ∘, convert # AbstractConstraint methods
using LinearAlgebra: LinearAlgebra, diag

# Basic functionality
include("./utils.jl")
export SuperDiagonal, abs_randn, all_recursive, eachfibre, getnotindex, geomean, identityslice, interlace, multifoldl, norm2, proj_one_hot, projsplx, proj_one_hot!, projsplx!
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
export NoConstraint, noconstraint
export ComposedConstraint

export ProjectedNormalization
export l2normalize!, l2normalize_rows!, l2normalize_cols!, l2normalize_1slices!, l2normalize_12slices!
export l1normalize!, l1normalize_rows!, l1normalize_cols!, l1normalize_1slices!, l1normalize_12slices!
export linftynormalize!, linftynormalize_rows!, linftynormalize_cols!, linftynormalize_1slices!, l1normalize_12slices!
export simplex!, simplex_rows!, simplex_cols!, simplex_1slices!, simplex_12slices!

export ScaledNormalization
export l2scale!, l2scale_rows!, l2scale_cols!, l2scale_1slices!, l2scale_12slices!
export l1scale!, l1scale_rows!, l1scale_cols!, l1scale_1slices!, l1scale_12slices!
export linftyscale!, linftyscale_rows!, linftyscale_cols!, linftyscale_1slices!, linftyscale_12slices!
export l1scale_average12slices!, l2scale_average12slices!, linftyscale_average12slices!

export Entrywise, IntervalConstraint
export nonnegative!, binary!, binaryproject
export l1norm, l2norm, linftynorm

export LinearConstraint

include("./stats.jl")
export AbstractStat
export DisplayDecomposition, EuclidianLipschitz, EuclidianStepSize, FactorNorms, GradientNorm, GradientNNCone
export IterateNormDiff, IterateRelativeDiff, Iteration, ObjectiveValue, ObjectiveRatio, PrintStats, RelativeError

include("./blockupdates.jl")
export AbstractStep
export LipschitzStep, ConstantStep, SPGStep

export AbstractUpdate, AbstractGradientDescent
export GradientDescent, BlockGradientDescent, MomentumUpdate

export ConstraintUpdate, GenericConstraintUpdate
export Projection, NNProjection, SafeNNProjection, Rescale

export BlockedUpdate
export updates
export smart_insert!, smart_interlase!, group_by_factor

#export block_gradient_decent, nn_block_gradient_decent, scale_nn_block_gradient_decent, proj_nn_block_gradient_decent

# High level / user-interface
include("./factorize.jl")
export factorize

include("./multiscale.jl")
export multiscale_factorize
export coarsen, interpolate, linear_smooth, scale_constraint

end # module Core
