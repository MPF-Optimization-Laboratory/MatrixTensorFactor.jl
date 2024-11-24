module DensityEstimationTools

using Statistics: mean, median, quantile, std
using KernelDensity

export DEFAULT_ALPHA, DEFAULT_N_SAMPLES
export default_bandwidth, make_densities, make_densities2d, standardize_KDEs, standardize_2d_KDEs, filter_inner_percentile, filter_2d_inner_percentile # Functions
export repeatcoord, kde2d, coordzip # 2d density estimation functions

include("densityestimation.jl")
include("densityestimation2d.jl")

end # module DensityEstimationTools
