"""
Matrix-Tensor Factorization
"""
module MatrixTensorFactor
using LinearAlgebra: norm, opnorm, Symmetric, ⋅
using Statistics: mean, median, quantile, std
using Random: randn
using Tullio: @einsum
using KernelDensity

# Method extentions
using Base: *

export Abstract3Tensor # Types
export combined_norm, dist_to_Ncone, nnmtf, rel_error, mean_rel_error, relative_error, slicewise_dot # Functions
export d_dx, d2_dx2, curvature, standard_curvature # Approximations
export nnmtf_proxgrad_online

export DEFAULT_ALPHA, DEFAULT_N_SAMPLES, MIN_STEP, MAX_STEP # Constants
export IMPLIMENTED_OPTIONS, IMPLIMENTED_NORMALIZATIONS, IMPLIMENTED_PROJECTIONS, IMPLIMENTED_CRITERIA, IMPLIMENTED_STEPSIZES # implimented options
export default_bandwidth, make_densities, make_densities2d, standardize_KDEs, standardize_2d_KDEs, filter_inner_percentile, filter_2d_inner_percentile # Functions
export repeatcoord, kde2d, coordzip # 2d density estimation functions

include("utils.jl")
include("matrixtensorfactorize.jl")
include("densityestimation.jl")
include("densityestimation2d.jl")

end
