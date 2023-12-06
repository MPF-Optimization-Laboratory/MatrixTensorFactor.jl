"""
Holds functions relevent for making 2D kernel density estimation
"""

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
