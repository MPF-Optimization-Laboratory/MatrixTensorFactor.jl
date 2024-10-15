"""Short helpers and operations for MTF.jl"""

"""
    Base.*(A::AbstractMatrix, B::AbstractArray)

Computes the AbstractArray C where ``C_{i_1 i_2 \\dots i_d} = \\sum_{l=1}^L A_{i_1 l} * B_{l i_2 \\dots i_d}``.

This is equivilent to the ``1``-mode product ``B \\times_1 A``.

Generalizes @einsum C[i,j,k] := A[i,l] * B[l,j,k].
"""
function Base.:*(A::AbstractMatrix, B::AbstractArray)
    sizeB = size(B)
    Bmat = reshape(B, sizeB[1], :)
    Cmat = A * Bmat
    C = reshape(Cmat, size(A)[1], sizeB[2:end]...)
    return C
end

"""
    slicewise_dot(A::AbstractArray, B::AbstractArray)

Constracts all but the first dimentions of A and B by performing a dot product over each `dim=1` slice.

Generalizes `@einsum C[s,r] := A[s,j,k]*B[r,j,k]` to arbitrary dimentions.
"""
function slicewise_dot(A::AbstractArray, B::AbstractArray)
    C = zeros(size(A)[1], size(B)[1]) # Array{promote_type(T, U), 2}(undef, size(A)[1], size(B)[1]) doesn't seem to be faster

    if A === B # use the faster routine if they are the same array
        return _slicewise_self_dot!(C, A)
    end

    for (i, A_slice) in enumerate(eachslice(A, dims=1))
        for (j, B_slice) in enumerate(eachslice(B, dims=1))
            C[i, j] = A_slice ⋅ B_slice
        end
    end
    return C
end

function _slicewise_self_dot!(C, A)
    enumerated_A_slices = enumerate(eachslice(A, dims=1))
    for (i, Ai_slice) in enumerated_A_slices
        for (j, Aj_slice) in enumerated_A_slices
            if i > j
                continue
            else
                C[i, j] = Ai_slice ⋅ Aj_slice
            end
        end
    end
    return Symmetric(C)
end

"""
    combined_norm(u, v, ...)

Compute the combined norm of the arguments as if all arguments were part of one large array.

This is equivilent to `norm(cat(u, v, ...))`, but this
implimentation avoids creating an intermediate array.

```jldoctest
u = [3 0]
v = [0 4 0]
combined_norm(u, v)

# output

5.0
```
"""
combined_norm(vargs...) = sqrt(sum(_norm2, vargs))
_norm2(x) = norm(x)^2

"""
    ReLU(x)

Rectified linear unit; takes the max of 0 and x.
"""
ReLU(x) = max(0,x)

"""
    rel_error(x, xhat)

Compute the relative error between x (true value) and xhat (its approximation).

The relative error is given by:
```math
\\frac{\\lVert \\hat{x} - x \\rVert}{\\lVert x \\rVert}
```
See also [`mean_rel_error`](@ref).
"""
function rel_error(xhat, x)
    return norm(xhat - x) / norm(x)
end

"""
    mean_rel_error(X, Xhat; dims=(1,2))

Compute the mean relative error between the dims-order slices of X and Xhat.

The mean relative error is given by:
```math
\\frac{1}{N}\\sum_{j=1}^N\\frac{\\lVert \\hat{X}_j - X_j \\rVert}{\\lVert X_j \\rVert}
```
See also [`rel_error`](@ref).
"""
function mean_rel_error(Xhat, X; dims=(1,2))
    hatslices = eachslice(Xhat; dims)
    slices = eachslice(X; dims)
    return mean(@. norm(hatslices - slices) / (norm(slices)))
end

"""
    d2_dx2(y::AbstractVector{<:Real}; order::Integer=length(y))

Approximates the 2nd derivative of a function using only given samples y of that function.

Assumes y came from f(x) where x was an evenly sampled, unit intervel grid.
Note the approximation uses centered three point finite differences for the
next-to-end points, and foward/backward three point differences for the begining/end points
respectively. The remaining interior points use five point differences.

Will use the largest order method possible by defult (currently 5 points), but can force
a specific order method with the keyword `order`.
See [`d_dx`](@ref).
"""
function d2_dx2(y::AbstractVector{<:Real}; order::Integer=length(y))
    n = length(y)
    if n < 3
        throw(ArgumentError("input $y must have at least length 3"))
    elseif order <= 2
        throw(ArgumentError("order $order must be at least 3"))
    end

    if order == 3
        return _d2_dx2_3(y)
    elseif order == 4
        return _d2_dx2_4(y)
    elseif order >= 5
        return _d2_dx2_5(y)
    else
        throw(ErrorException("Something went wrong with the order $order."))
    end
end
# TODO is there a package that does this? The ones I've seen require the forward function.

function _d2_dx2_3(y::AbstractVector{<:Real})
    d = similar(y)
    for i in eachindex(y)[begin+1:end-1]
        d[i] = y[i-1] - 2*y[i] + y[i+1]
    end
    # Assume the same curvature at the end points
    d[begin] = d[begin+1]
    d[end] = d[end-1]
    return d
end

function _d2_dx2_4(y::AbstractVector{<:Real})
    d = similar(y)
    each_i = eachindex(y)

    for i in each_i[begin+1:end-1]
        d[i] = y[i-1] - 2*y[i] + y[i+1] # Note this is the same estimate as _d2_dx2_3
    end

    # Four point forward/backwards estimate at end points
    i = each_i[begin]
    d[i] = (6*y[i] - 15*y[i+1] + 12*y[i+2] - 3*y[i+3])/3

    i = each_i[end]
    d[i] = (-3*y[i-3] + 12*y[i-2] -15*y[i-1] +6*y[i])/3
    return d
end

function _d2_dx2_5(y::AbstractVector{<:Real})
    d = similar(y)
    each_i = eachindex(y)

    # Interior Ppints
    for i in each_i[begin+2:end-2]
        d[i] = (-y[i-2] + 16*y[i-1] - 30*y[i] + 16*y[i+1] - y[i+2])/12
    end

    # Boundary and next-to boundary points
    i = each_i[begin+2]
    d[i-2] = (35*y[i-2] - 104*y[i-1] + 114*y[i] - 56*y[i+1] + 11*y[i+2])/12
    d[i-1] = (11*y[i-2] - 20*y[i-1] + 6*y[i] + 4*y[i+1] - y[i+2])/12

    i = each_i[end-2]
    d[i+1] = (-y[i-2] + 4*y[i-1] + 6*y[i] - 20*y[i+1] + 11*y[i+2])/12
    d[i+2] = (11*y[i-2] - 56*y[i-1] + 114*y[i] - 104*y[i+1] + 35*y[i+2])/12

    return d
end

"""
    d_dx(y::AbstractVector{<:Real})

Approximates the 1nd derivative of a function using only given samples y of that function.

Assumes y came from f(x) where x was an evenly sampled, unit intervel grid.
Note the approximation uses centered three point finite differences for the
next-to-end points, and foward/backward three point differences for the begining/end points
respectively. The remaining interior points use five point differences.

Will use the largest order method possible by defult (currently 5 points), but can force
a specific order method with the keyword `order`.
See [`d2_dx2`](@ref).
"""
function d_dx(y::AbstractVector{<:Real}; order::Integer=length(y))
    n = length(y)
    if n < 3
        throw(ArgumentError("input $y must have at least length 3"))
    elseif order <= 2
        throw(ArgumentError("order $order must be at least 3"))
    end

    if order == 3
        return _d_dx_3(y)
    elseif order == 4
        return _d_dx_4(y)
    elseif order >= 5
        return _d_dx_5(y)
    else
        throw(ErrorException("Something went wrong with the order $order."))
    end
end

function _d_dx_3(y::AbstractVector{<:Real})
    d = similar(y)
    each_i = eachindex(y)

    for i in each_i[begin+1:end-1]
        d[i] = (-y[i-1] + y[i+1])/2
    end

    i = each_i[begin+1]
    d[begin] = (-3*y[i-1] + 4*y[i] - y[i+1])/2

    i = each_i[end-1]
    d[end] = (y[i-1] - 4*y[i] + 3*y[i+1])/2
    return d
end

function _d_dx_4(y::AbstractVector{<:Real})
    d = similar(y)
    each_i = eachindex(y)

    for i in each_i[begin+1:end-2]
        d[i] = (-2*y[i-1] - 3*y[i] + 6*y[i+1] - y[i+2])/6 # four points is odd so using a half forward estimate
    end

    i = each_i[begin]
    d[i] = (-11*y[i] + 18*y[i+1] - 9*y[i+2] + 2*y[i+3])/6

    i = each_i[end]
    d[i-1] = (y[i-3] - 6*y[i-2] + 3*y[i-1] + 2*y[i])/6
    d[i]   = (-2*y[i-3] + 9*y[i-2] - 18*y[i-1] + 11*y[i])/6

    return d
end

function _d_dx_5(y::AbstractVector{<:Real})
    d = similar(y)
    each_i = eachindex(y)

    # Interior Ppints
    for i in each_i[begin+2:end-2]
        d[i] = (2*y[i-2] - 16*y[i-1] + 16*y[i+1] - 2*y[i+2])/24
    end

    # Boundary and next-to boundary points
    i = each_i[begin+2]
    d[i-2] = (-50*y[i-2] + 96*y[i-1] - 72*y[i] + 32*y[i+1] - 6*y[i+2])/24
    d[i-1] = (-6*y[i-2] - 20*y[i-1] + 36*y[i] - 12*y[i+1] + 2*y[i+2])/24

    i = each_i[end-2]
    d[i+1] = (-2*y[i-2] + 12*y[i-1] - 36*y[i] + 20*y[i+1] + 6*y[i+2])/24
    d[i+2] = (6*y[i-2] - 32*y[i-1] + 72*y[i] - 96*y[i+1] + 50*y[i+2])/24

    return d
end

"""
    curvature(y::AbstractVector{<:Real})

Approximates the signed curvature of a function given evenly spaced samples.

Uses [`d_dx`](@ref) and [`d2_dx2`](@ref) to approximate the first two derivatives.
"""
function curvature(y::AbstractVector{<:Real}; kwargs...)
    dy_dx = d_dx(y; kwargs...)
    dy2_dx2 = d2_dx2(y; kwargs...)
    return @. dy2_dx2 / (1 + dy_dx^2)^1.5
end

"""
    standard_curvature(y::AbstractVector{<:Real})

Approximates the signed curvature of a function, scaled to the unit box ``[0,1]^2``.

See [`curvature`](@ref).
"""
function standard_curvature(y::AbstractVector{<:Real}; kwargs...)
    Δx = 1 / (length(y) - 1) # An interval 0:10 has length(0:10) = 11, but measure 10-0 = 10
    y_max = maximum(y)
    dy_dx = d_dx(y; kwargs...) / (Δx * y_max)
    dy2_dx2 = d2_dx2(y; kwargs...) / (Δx^2 * y_max)
    return @. dy2_dx2 / (1 + dy_dx^2)^1.5
end

"""

    projsplx(y::AbstractVector{<:Real})

Projects (in Euclidian distance) the vector y into the simplex.

[1] Yunmei Chen and Xiaojing Ye, "Projection Onto A Simplex", 2011
"""
function projsplx(y)
    n = length(y)

    if n==1 # quick exit for trivial length-1 "vectors" (i.e. scalars)
        return [one(eltype(y))]
    end

    y_sorted = sort(y[:]) # Vectorize/extract input and sort all entries
    i = n - 1
    t = 0 # need to ensure t has scope outside the while loop
    while true
        t = (sum(@view y_sorted[i+1:end]) - 1) / (n-i)
        if t >= y_sorted[i]
            break
        else
            i -= 1
        end

        if i >= 1
            continue
        else # i == 0
            t = (sum(y_sorted) - 1) / n
            break
        end
    end
    return ReLU.(y .- t)
end

"""
Finds the radius of the circumscribed circle between points (a,f), (b,g), (c,h)
"""
function circumscribed_radius((a,f),(b,g),(c,h))
    d = 2*(a*(g-h)+b*(h-f)+c*(f-g))
    p = ((a^2+f^2)*(g-h)+(b^2+g^2)*(h-f)+(c^2+h^2)*(f-g)) / d
    q = ((a^2+f^2)*(b-c)+(b^2+g^2)*(c-a)+(c^2+h^2)*(a-b)) / d
    r = sqrt((a-p)^2+(f-q)^2)
    return r
end

function circumscribed_standard_curvature(y)
    n = length(v)
    ymax = maximum(y)
    y = y / ymax
    k = zero(ymax)
    a, b, c = 0, 1/n, 2/n
    for i in eachindex(k)[2:end-1]
        k[i] = 1 / circumscribed_radius((a,y[i-1]),(b,y[i]),(c,y[i+1]))
    end
    k[1] = k[2]
    k[end] = k[end-1]
    return k
end
