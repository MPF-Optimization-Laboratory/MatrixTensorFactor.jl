"""
Matrix-Tensor Factorization
"""
module MatrixTensorFactor

using Random: randn

# include("../Core/Core.jl")
using ..Core # note the two dots .. since the module Core is not in the same folder as MatrixTensorFactor

# include("../Core/utils.jl")
# include("../Core/tensorproducts.jl")
# include("../Core/decomposition.jl")
# include("../Core/objective.jl")
# include("../Core/constraint.jl")
# include("../Core/stats.jl")
# include("../Core/blockupdates.jl")
# include("../Core/factorize.jl")

export nnmtf, nnmtf_proxgrad_online # Functions
export d_dx, d2_dx2, curvature, standard_curvature # Approximations

export IMPLIMENTED_OPTIONS, IMPLIMENTED_NORMALIZATIONS, IMPLIMENTED_METRICS
export IMPLIMENTED_PROJECTIONS, IMPLIMENTED_CRITERIA, IMPLIMENTED_STEPSIZES # implimented options

include("curvaturetools.jl")
include("nnmtf.jl")

end
