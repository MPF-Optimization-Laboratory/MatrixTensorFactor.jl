"""
Filters elements so only the ones in the inner P percentile remain.
"""
filter_inner_percentile(v, P) = filter(_inrange(v, P), v)

"""
Returns a function that checks if a value is in the inner P percentile of the values in v.
"""
function _inrange(v, P)
    p_low = (100 - P) / 2
    p_high = 100 - p_low
    a, b = quantile(v, [p_low, p_high] ./ 100)
    return x -> (a ≤ x ≤ b)
end

"""
    DEFAULT_ALPHA = 0.9::Real

Smoothing parameter for calculating a kernel's bandwidth.
"""
global DEFAULT_ALPHA = 0.9::Real

"""
    default_bandwidth(data; alpha=0.9, inner_percentile=100)

Coppied from KernelDensity since this function is not exported. I want access
to it so that the same bandwidth can be used for different densities for the
same measurements.
"""
function default_bandwidth(
    data,#::AbstractVector{<: Real},
    alpha::Real = DEFAULT_ALPHA,
    inner_percentile::Integer=100,
    )

    # Filter outliers, remove values outside the inner percentile
    if inner_percentile < 100
        data = filter_inner_percentile(data, inner_percentile)
    end

    # Determine length of data
    ndata = length(data)
    ndata <= 1 && return alpha

    # Calculate width using variance and IQR
    var_width = std(data)
    q25, q75 = quantile(data, [0.25, 0.75])
    quantile_width = (q75 - q25) / 1.34

    # Deal with edge cases with 0 IQR or variance
    width = min(var_width, quantile_width)
    if width == 0.0
        if var_width == 0.0
            width = 1.0
        else
            width = var_width
        end
    end

    # Set bandwidth using Silverman's rule of thumb
    return alpha * width * ndata^(-0.2)
end

"""
    make_densities(s::Sink; kwargs...)
    make_densities(s::Sink, domains::AbstractVector{<:AbstractVector}; kwargs...)

Estimates the densities for each measurement in a Sink.

When given domains, a list where each entry is a domain for a different measurement,
resample the kernel on this domain.

# Parameters
- `bandwidths::AbstractVector{<:Real}`: list of bandwidths used for each measurement's
density estimation
- `inner_percentile::Integer=100`: value between 0 and 100 that filters out each measurement
by using the inner percentile range. This can help remove outliers and focus in on where the
bulk of the data is.

# Returns
- `density_estimates::Vector{UnivariateKDE}`
"""
function make_densities(
    data::AbstractVector{T};
    inner_percentile::Integer=100,
    #bandwidths::AbstractVector{<:Real}=default_bandwidth.(
    #    s,DEFAULT_ALPHA,inner_percentile),
    ) where T #<: AbstractVecOrMat
    # Argument Handeling: check inner_percentile is a percentile
    (0 < inner_percentile <= 100) ||
        ArgumentError("inner_percentile must be between 0 and 100, got $inner_percentile")

    # Loop setup
    n_measurements = length(data)
    density_estimates = Vector{Union{UnivariateKDE,BivariateKDE}}(undef, n_measurements)

    #for (i, (measurement_values, b)) in enumerate(zip(data, bandwidths))
    for (i, measurement_values) in enumerate(data)
        # Estimate density based on the inner precentile to ignore outliers
        #measurement_values = filter_inner_percentile(measurement_values, inner_percentile)
        density_estimates[i] = kde(measurement_values)#, bandwidth=b)
    end

    return density_estimates
end

#function make_densities(
#    sinks::Sink,
#    domains::AbstractVector{<:AbstractVector};
#    kwargs...
#    )
#    KDEs = make_densities(sinks; kwargs...)
#    KDEs_new = pdf.(KDEs, domains)
#    return KDEs_new
#end

"""
    DEFAULT_N_SAMPLES = 64::Integer

Number of samples to use when standardizing a vector of density estimates.
"""
const DEFAULT_N_SAMPLES = 64::Integer

"""
    standardize_KDEs(KDEs::AbstractVector{UnivariateKDE}; n_samples=DEFAULT_N_SAMPLES,)

Resample the densities so they all are smapled from the same domain.
"""
function standardize_KDEs(KDEs; n_samples=DEFAULT_N_SAMPLES,)
    a = minimum(d -> d.x[begin], KDEs) # smallest left endpoint
    b = maximum(d -> d.x[end]  , KDEs) # biggest right endpoint

    x_new = range(a, b, length=n_samples) # make the (larger) x-values range
    KDEs_new = pdf.(KDEs, (x_new,)) # Resample the densities on the new range.
                                    # Note the second argument is a 1-tuple so that we can
                                    # broadcast over the first argument only, i.e.
                                    # KDEs_new[i] = pdf(KDEs[i], x_new)
    return KDEs_new, x_new
end

"""
Resample the densities within each sink/source so that like-measurements use the same scale.
"""
function standardize_KDEs(
    KDEs_by_source::AbstractVector{<:AbstractVector{<:UnivariateKDE}};
    n_samples=DEFAULT_N_SAMPLES,
    )
    # Group same measurements from different sources
    KDEs_by_measurement = zip(KDEs_by_source...)

    # Standardize each measurement, use zip to unpack the (KDEs_new, x_new) pairs
    KDEs_by_measurement, xs = zip((standardize_KDEs.(KDEs_by_measurement; n_samples))...)

    # Group different measurements from the same source back together
    KDEs_by_source = zip(KDEs_by_measurement...)

    return collect(KDEs_by_source), collect(xs)
end

# """Custom zip to use vectors rather than tuples as the container"""
# function myzip(list_of_lists)
#     return [[list[i] for list in list_of_lists] for i in eachindex(list_of_lists[begin])]
# end
