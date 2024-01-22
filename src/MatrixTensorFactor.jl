"""
Matrix-Tensor Factorization
"""
module MatrixTensorFactor
using LinearAlgebra: norm, opnorm
#using Plots: plot, heatmap
using Statistics: mean, median, quantile, std
using Random: randn
using Einsum: @einsum
using KernelDensity

# Method extentions
using Base: *

export Abstract3Tensor # Types
export combined_norm, dist_to_Ncone, nnmtf, rel_error, mean_rel_error, residual # Functions
export d_dx, d2_dx2, curvature, standard_curvature # Approximations

export DEFAULT_ALPHA, DEFAULT_N_SAMPLES, MIN_STEP, MAX_STEP # Constants
export IMPLIMENTED_OPTIONS, IMPLIMENTED_NORMALIZATIONS, IMPLIMENTED_PROJECTIONS, IMPLIMENTED_CRITERIA, IMPLIMENTED_STEPSIZES # implimented options
export default_bandwidth, make_densities, standardize_KDEs, filter_inner_percentile # Functions
export repeatcoord, kde2d, coordzip # 2d density estimation functions

include("utils.jl")
include("matrixtensorfactorize.jl")
include("densityestimation.jl")
include("densityestimation2d.jl")

end
