"""Short helpers and operations related to finite differences and curavture"""

# """
#     d2_dx2(y::AbstractVector{<:Real}; order::Integer=length(y))

# Approximates the 2nd derivative of a function using only given samples y of that function.

# Assumes y came from f(x) where x was an evenly sampled, unit intervel grid.
# Note the approximation uses centered three point finite differences for the
# next-to-end points, and foward/backward three point differences for the begining/end points
# respectively. The remaining interior points use five point differences.

# Will use the largest order method possible by defult (currently 5 points), but can force
# a specific order method with the keyword `order`.
# See [`d_dx`](@ref).
# """
# function d2_dx2(y::AbstractVector{<:Real}; order::Integer=length(y))
#     n = length(y)
#     if n < 3
#         throw(ArgumentError("input $y must have at least length 3"))
#     elseif order <= 2
#         throw(ArgumentError("order $order must be at least 3"))
#     end

#     if order == 3
#         return _d2_dx2_3(y)
#     elseif order == 4
#         return _d2_dx2_4(y)
#     elseif order >= 5
#         return _d2_dx2_5(y)
#     else
#         throw(ErrorException("Something went wrong with the order $order."))
#     end
# end
# # TODO is there a package that does this? The ones I've seen require the forward function.

# function _d2_dx2_3(y::AbstractVector{<:Real})
#     d = similar(y)
#     for i in eachindex(y)[begin+1:end-1]
#         d[i] = y[i-1] - 2*y[i] + y[i+1]
#     end
#     # Assume the same curvature at the end points
#     d[begin] = d[begin+1]
#     d[end] = d[end-1]
#     return d
# end

# function _d2_dx2_4(y::AbstractVector{<:Real})
#     d = similar(y)
#     each_i = eachindex(y)

#     for i in each_i[begin+1:end-1]
#         d[i] = y[i-1] - 2*y[i] + y[i+1] # Note this is the same estimate as _d2_dx2_3
#     end

#     # Four point forward/backwards estimate at end points
#     i = each_i[begin]
#     d[i] = (6*y[i] - 15*y[i+1] + 12*y[i+2] - 3*y[i+3])/3

#     i = each_i[end]
#     d[i] = (-3*y[i-3] + 12*y[i-2] -15*y[i-1] +6*y[i])/3
#     return d
# end

# function _d2_dx2_5(y::AbstractVector{<:Real})
#     d = similar(y)
#     each_i = eachindex(y)

#     # Interior Ppints
#     for i in each_i[begin+2:end-2]
#         d[i] = (-y[i-2] + 16*y[i-1] - 30*y[i] + 16*y[i+1] - y[i+2])/12
#     end

#     # Boundary and next-to boundary points
#     i = each_i[begin+2]
#     d[i-2] = (35*y[i-2] - 104*y[i-1] + 114*y[i] - 56*y[i+1] + 11*y[i+2])/12
#     d[i-1] = (11*y[i-2] - 20*y[i-1] + 6*y[i] + 4*y[i+1] - y[i+2])/12

#     i = each_i[end-2]
#     d[i+1] = (-y[i-2] + 4*y[i-1] + 6*y[i] - 20*y[i+1] + 11*y[i+2])/12
#     d[i+2] = (11*y[i-2] - 56*y[i-1] + 114*y[i] - 104*y[i+1] + 35*y[i+2])/12

#     return d
# end

# """
#     d_dx(y::AbstractVector{<:Real})

# Approximates the 1nd derivative of a function using only given samples y of that function.

# Assumes y came from f(x) where x was an evenly sampled, unit intervel grid.
# Note the approximation uses centered three point finite differences for the
# next-to-end points, and foward/backward three point differences for the begining/end points
# respectively. The remaining interior points use five point differences.

# Will use the largest order method possible by defult (currently 5 points), but can force
# a specific order method with the keyword `order`.
# See [`d2_dx2`](@ref).
# """
# function d_dx(y::AbstractVector{<:Real}; order::Integer=length(y))
#     n = length(y)
#     if n < 3
#         throw(ArgumentError("input $y must have at least length 3"))
#     elseif order <= 2
#         throw(ArgumentError("order $order must be at least 3"))
#     end

#     if order == 3
#         return _d_dx_3(y)
#     elseif order == 4
#         return _d_dx_4(y)
#     elseif order >= 5
#         return _d_dx_5(y)
#     else
#         throw(ErrorException("Something went wrong with the order $order."))
#     end
# end

# function _d_dx_3(y::AbstractVector{<:Real})
#     d = similar(y)
#     each_i = eachindex(y)

#     for i in each_i[begin+1:end-1]
#         d[i] = (-y[i-1] + y[i+1])/2
#     end

#     i = each_i[begin+1]
#     d[begin] = (-3*y[i-1] + 4*y[i] - y[i+1])/2

#     i = each_i[end-1]
#     d[end] = (y[i-1] - 4*y[i] + 3*y[i+1])/2
#     return d
# end

# function _d_dx_4(y::AbstractVector{<:Real})
#     d = similar(y)
#     each_i = eachindex(y)

#     for i in each_i[begin+1:end-2]
#         d[i] = (-2*y[i-1] - 3*y[i] + 6*y[i+1] - y[i+2])/6 # four points is odd so using a half forward estimate
#     end

#     i = each_i[begin]
#     d[i] = (-11*y[i] + 18*y[i+1] - 9*y[i+2] + 2*y[i+3])/6

#     i = each_i[end]
#     d[i-1] = (y[i-3] - 6*y[i-2] + 3*y[i-1] + 2*y[i])/6
#     d[i]   = (-2*y[i-3] + 9*y[i-2] - 18*y[i-1] + 11*y[i])/6

#     return d
# end

# function _d_dx_5(y::AbstractVector{<:Real})
#     d = similar(y)
#     each_i = eachindex(y)

#     # Interior Ppints
#     for i in each_i[begin+2:end-2]
#         d[i] = (2*y[i-2] - 16*y[i-1] + 16*y[i+1] - 2*y[i+2])/24
#     end

#     # Boundary and next-to boundary points
#     i = each_i[begin+2]
#     d[i-2] = (-50*y[i-2] + 96*y[i-1] - 72*y[i] + 32*y[i+1] - 6*y[i+2])/24
#     d[i-1] = (-6*y[i-2] - 20*y[i-1] + 36*y[i] - 12*y[i+1] + 2*y[i+2])/24

#     i = each_i[end-2]
#     d[i+1] = (-2*y[i-2] + 12*y[i-1] - 36*y[i] + 20*y[i+1] + 6*y[i+2])/24
#     d[i+2] = (6*y[i-2] - 32*y[i-1] + 72*y[i] - 96*y[i+1] + 50*y[i+2])/24

#     return d
# end

"""
    d_dx(y::AbstractVector{<:Real})

Approximate first derivative with finite elements. Assumes y[i] = y(x_i) are samples with unit spaced inputs x_{i+1} - x_i = 1.
"""
function d_dx(y::AbstractVector{<:Real})
    if length(y) < 3
        throw(ArgumentError("y must have length at least 3, got $(length(y))"))
    end

    d = similar(y)
    each_i = eachindex(y)

    # centred estimate
    for i in each_i[begin+1:end-1]
        d[i] = (-y[i-1] + y[i+1])/2
    end

    # three point forward/backward estimate
    i = each_i[begin+1]
    d[begin] = (-3*y[i-1] + 4*y[i] - y[i+1])/2

    i = each_i[end-1]
    d[end] = (y[i-1] - 4*y[i] + 3*y[i+1])/2
    return d
end

"""
    d2_dx2(y::AbstractVector{<:Real})

Approximate second derivative with finite elements. Assumes y[i] = y(x_i) are samples with unit spaced inputs x_{i+1} - x_i = 1.
"""
function d2_dx2(y::AbstractVector{<:Real})
    if length(y) < 3
        throw(ArgumentError("y must have length at least 3, got $(length(y))"))
    end

    d = similar(y)
    for i in eachindex(y)[begin+1:end-1]
        d[i] = y[i-1] - 2*y[i] + y[i+1]
    end
    # Assume the same second derivative at the end points
    d[begin] = d[begin+1]
    d[end] = d[end-1]
    return d
end

function cubic_spline_coefficients(y::AbstractVector{<:Real}; h=1)
    # Set up variables
    n = length(y)
    T = eltype(y)
    f = diff([y; y[end]]) # use diff([y; zero(T)]) to clamp at a y value of 0 instead of a repeated boundary condition

    # solve the system Mb=v
    M = spline_mat(n)
    v = 3/h^2 .* [1 - 2y[1] + y[2]; diff(f); 0]
    b = M \ v

    # use b to find the other coefficients
    c = [f[i]/h - h/3*(b[i+1] + 2b[i]) for i in 1:n]
    a = diff(b) ./ 3h
    d = copy(y)

    # truncate b from length n+1 to n
    return a, b[1:end-1], c, d
end

function make_spline(y::AbstractVector{<:Real}; h=1)
    a, b, c, d = cubic_spline_coefficients(y::AbstractVector{<:Real}; h=1)
    n = length(y)

    function f(x)
        i = Int(floor(x))

        # find which spline piece to use
        # extrapolating from the first or last spline if needed
        if i < 1
            i = 1
        elseif i > n
            i = n
        end

        h = x - i

        return a[i]*h^3 + b[i]*h^2 + c[i]*h + d[i]
    end

    return f
end

function d_dx_and_d2_dx2_spline(y::AbstractVector{<:Real}; h=1)
    _, b, c, _ = cubic_spline_coefficients(y::AbstractVector{<:Real}; h=1)
    dy_dx = c
    dy2_dx2 = 2b
    return dy_dx, dy2_dx2
end

function spline_mat(n)
    du = [0; ones(Int, n-1)]
    dd = [6; 4*ones(Int, n-1) ; 1]
    dl = [ones(Int, n-1); 0]

    return Tridiagonal(dl, dd, du)
end

"""
    curvature(y::AbstractVector{<:Real})

Approximates the signed curvature of a function given evenly spaced samples.

Uses [`d_dx`](@ref) and [`d2_dx2`](@ref) to approximate the first two derivatives.
"""
function curvature(y::AbstractVector{<:Real}; method=:finite_differences, kwargs...)
    if method == finite_differences
        dy_dx = d_dx(y; kwargs...)
        dy2_dx2 = d2_dx2(y; kwargs...)
        return @. dy2_dx2 / (1 + dy_dx^2)^1.5
    elseif method == :splines
        dy_dx, dy2_dx2 = d_dx_and_d2_dx2_spline(y; h=1)
        return @. dy2_dx2 / (1 + dy_dx^2)^1.5
    else
        throw(ArgumentError("method $method not implemented"))
    end
end

"""
    standard_curvature(y::AbstractVector{<:Real})

Approximates the signed curvature of a function, scaled to the unit box ``[0,1]^2``.

See [`curvature`](@ref).
"""
function standard_curvature(y::AbstractVector{<:Real}; method=:finite_differences, kwargs...)
    Δx = 1 / (length(y) - 1) # An interval 0:10 has length(0:10) = 11, but measure 10-0 = 10
    if method == finite_differences
        y_max = maximum(y)
        dy_dx = d_dx(y; kwargs...) / (Δx * y_max)
        dy2_dx2 = d2_dx2(y; kwargs...) / (Δx^2 * y_max)
        return @. dy2_dx2 / (1 + dy_dx^2)^1.5
    elseif method == :splines
        # y_max = 1
        dy_dx, dy2_dx2 = d_dx_and_d2_dx2_spline(y; h=Δx)
        return @. dy2_dx2 / (1 + dy_dx^2)^1.5
    else
        throw(ArgumentError("method $method not implemented"))
    end
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
