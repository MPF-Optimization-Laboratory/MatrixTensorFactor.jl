"""
Matrix-Tensor Factorization
"""
module MatrixTensorFactor
using LinearAlgebra: norm
using Plots: plot
using Statistics: mean, median
using Random: randn
using Einsum: @einsum
using KernelDensity

# Method extentions
using Base: *

export Abstract3Tensor # Types
export combined_norm, dist_to_Ncone, nnmtf, nnmtf2d, plot_factors, rel_error, mean_rel_error # Functions
export d_dx, d2_dx2, curvature, standard_curvature # Approximations

export DEFAULT_ALPHA, DEFAULT_N_SAMPLES # Constants
export default_bandwidth, make_densities, standardize_KDEs # Functions
export repeatcoord, kde2d, coordzip # 2d density estimation functions

include("utils.jl")
include("matrixtensorfactorize.jl")
include("matrixtensorfactorize2d.jl")
include("densityestimation.jl")
include("densityestimation2d.jl")

end
