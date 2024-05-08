"""
Holds functions relevent for making 2D kernel density estimation
"""

"""
Filters 2d elements so only the ones in the inner P percentile remain. See [`filter_inner_percentile`](@ref).
"""
filter_2d_inner_percentile(vs, P) = filter(_in2drange(vs, P), vs)

"""
Returns a function that checks if each coordinate is in the inner P percentile of the values in vs.
"""
function _in2drange(vs, P)
    p_low = (100 - P) / 2
    p_high = 100 - p_low
    a, b = quantile([v[1] for v in vs], [p_low, p_high] ./ 100)
    c, d = quantile([v[2] for v in vs], [p_low, p_high] ./ 100)
    return x -> ((a ≤ x[1] ≤ b) && (c ≤ x[2] ≤ d))
end

# TODO extend this to arbitrary number of dimentions

"""
make_densities2d(s::Sink; kwargs...)
make_densities2d(s::Sink, domains::AbstractVector{<:AbstractVector}; kwargs...)

Similar to [`make_densities`](@ref) but performs the KDE on 2 measurements jointly.
"""
function make_densities2d(
data::AbstractVector{T};
inner_percentile::Integer=100,
#bandwidths::AbstractVector{<:Real}=default_bandwidth.(
#    collect(eachmeasurement(s)),DEFAULT_ALPHA,inner_percentile),
) where T
# Argument Handeling: check inner_percentile is a percentile
(0 < inner_percentile <= 100) ||
    ArgumentError("inner_percentile must be between 0 and 100, got $inner_percentile")

#(length(data[begin]) == 2) ||
#    ArgumentError("should only be 2 measurements for the grain in s, got $length(getmeasurements(s))")

#data = filter_2d_inner_percentile(data)

KDE = kde(hcat(collect(array(g) for g in data)...)'; bandwidth=tuple(bandwidths...))
return KDE
end

"""
    standardize_2d_KDEs(KDEs::AbstractVector{BivariateKDE}; n_samples=DEFAULT_N_SAMPLES,)

Resample the densities so they all are sampled from the same x and y coordinates.
"""
function standardize_2d_KDEs(KDEs; n_samples=DEFAULT_N_SAMPLES,)
    a = minimum(f -> f.x[begin], KDEs) # smallest left endpoint
    b = maximum(f -> f.x[end]  , KDEs) # biggest right endpoint
    c = minimum(f -> f.y[begin], KDEs) # smallest left endpoint
    d = maximum(f -> f.y[end]  , KDEs) # biggest right endpoint

    x_new = range(a, b, length=n_samples) # make the (larger) x-values range
    y_new = range(c, d, length=n_samples) # make the (larger) y-values range
    KDEs_new = pdf.(KDEs, (x_new,), (y_new,)) # Resample the densities on the new domain.
                                    # Note the second argument is a 1-tuple so that we can
                                    # broadcast over the first argument only, i.e.
                                    # KDEs_new[i] = pdf(KDEs[i], x_new)
    return KDEs_new, x_new, y_new
end


"""
    repeatcoord(coordinates, values)

Repeates coordinates the number of times given by values.

Both lists should be the same length.

Example
-------
coordinates = [(0,0), (1,1), (1,2)]
values = [1, 3, 2]
repeatcoord(coordinates, values)

[(0,0), (1,1), (1,1), (1,1), (1,2), (1,2)]
"""
function repeatcoord(coordinates, values)
    vcat(([coord for _ in 1:v] for (coord, v) in zip(coordinates, values))...)
end

"""
    kde2d((xs, ys), values)

Performs a 2d KDE based on two lists of coordinates, and the value at those coordinates.
Input
-----
- `xs, ys::Vector{Real}`: coordinates/locations of samples
- `values::Vector{Integer}`: value of the sample
Returns
-------
- `f::BivariateKDE` use f.x, f.y for the location of the (re)sampled KDE,
and f.density for the sample values of the KDE
"""
function kde2d((xs, ys), values)
    xsr, ysr = [repeatcoord(coord, values) for coord in (xs, ys)]
    coords = hcat(xsr, ysr)
    f = kde(coords)
    return f
end

"""
    coordzip(rcoords)

Zips the "x" and "y" values together into a list of x coords and y coords.
Example
-------
coordzip([(0,0), (1,1), (1,1), (1,1), (1,2), (1,2)])

[[0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 2, 2]]
"""
function coordzip(rcoords)
    [[x for x in xs] for xs in zip(rcoords...)]
end
