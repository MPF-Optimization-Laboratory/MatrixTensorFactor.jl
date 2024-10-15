"""
Matrix-Tensor Factorization
"""
module MatrixTensorFactor

using Random: randn

include("../Core/Core.jl")
using .Core

export nnmtf, nnmtf_proxgrad_online # Functions
export d_dx, d2_dx2, curvature, standard_curvature # Approximations

export IMPLIMENTED_OPTIONS, IMPLIMENTED_NORMALIZATIONS, IMPLIMENTED_METRICS
export IMPLIMENTED_PROJECTIONS, IMPLIMENTED_CRITERIA, IMPLIMENTED_STEPSIZES # implimented options

include("curvaturetools.jl")
include("nnmtf.jl")

end
