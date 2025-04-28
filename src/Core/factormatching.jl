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
        @warn "`match_slices!` may not have found the best ordering; using greedy approach."
        ordering = zeros(Int, n_slices)
        X_slices = collect(enumerate(eachslice(X; dims))) # [(1, X1), (2, X2), ...]
        for (j, Yj) ∈ enumerate(eachslice(Y; dims))
            _, idx = findmin(j_Xj -> dist(j_Xj[2], Yj), X_slices)
            j_match = X_slices[idx][1] # index of the slice: X_slices[idx] == (j, Xj)
            ordering[j] = j_match
            popat!(X_slices, idx) # remove slice Xj from the options
        end
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
