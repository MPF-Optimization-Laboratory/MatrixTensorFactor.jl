"""
    match_slices!(X, Y; dims, dist=L2)

Mutates X so the order-`dims` slices best match the slices of Y.
"""
function match_slices!(X::AbstractArray, Y::AbstractArray; dims, dist=L2)
    size(X) == size(Y) || ArgumentError("Shape of X, $(size(X)) does not match shape of Y, $(size(Y))")

    n_slices = size(X, dims)
    ordering = zeros(Int, n_slices)

    # for every slice of Y, find the best matching slice in X
    for (j, Yj) ∈ enumerate(eachslice(Y; dims))
        _, j_match = findmin(Xj -> dist(Xj, Yj), eachslice(X; dims))
        ordering[j] = j_match
    end

    if !allunique(ordering)
        # Global optimal order would involve checking factorial(n_slices) options
        # Use a greedy approach instead
        @warn "`match_slices!` may not have found the best ordering; using O(n²) greedy approach."

        dist_matrix = zeros(n_slices, n_slices)
        for (j, Yj) ∈ enumerate(eachslice(Y; dims))
            for (i, Xi) ∈ enumerate(eachslice(X; dims))
                dist_matrix[i, j] = dist(Xi, Yj)
            end
        end

        # Find the closest slices (Xi, Yj), remove the from the options, and repeat.
        # We don't saved a sliced dist_matrix since that would renumber the rows/cols.
        # Instead fill the row/col with Inf so argmin can no longer pick that row/col.
        pairings = zeros(CartesianIndex{2}, n_slices)
        for n in 1:n_slices
            pairing = argmin(dist_matrix) # CartesianIndex(i, j)
            dist_matrix[pairing[1], :] .= Inf # remove row i
            dist_matrix[:, pairing[2]] .= Inf # remove col j
            pairings[n] = pairing
        end

        sort!(pairings) # Sorts by second index (Yj)

        ordering = map(x->x[1], pairings) # Ordering is given by the first index (Xi)
    end

    # Permute slice of X
    reslice = Any[(:) for _ in 1:ndims(X)]
    reslice[dims] = ordering
    X .= @view X[reslice...]

    return ordering
end

"""
    match_rows!(X::AbstractMatrix, Y::AbstractMatrix; kwargs...)

`match_slices!` along the first dimension.
"""
match_rows!(X::AbstractMatrix, Y::AbstractMatrix; kwargs...) = match_slices!(X, Y; dims=1, kwargs...)

"""
    match_cols!(X::AbstractMatrix, Y::AbstractMatrix; kwargs...)

`match_slices!` along the second dimension.
"""
match_cols!(X::AbstractMatrix, Y::AbstractMatrix; kwargs...) = match_slices!(X, Y; dims=2, kwargs...)
