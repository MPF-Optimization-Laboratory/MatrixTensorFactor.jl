"""
Script for comparing multiscale to single scale solving on the following problem for a given vector y and matrix A.

minₓ 0.5||Ax - y||₂² s.t. ||x||₁ = 1, and x ≥ 0
"""

using BlockTensorFactorization
using Random
using LinearAlgebra
using Plots

n_scales = 9
n_moments = 10
fine_scale_size = 2^n_scales + 1
scale_to_skip(s) = 2^(s-1)
# t = range(0, 1, length=fine_scale_size)
h(t) = -84t^4 + 146.4t^3 - 74.4t^2 + 12t
f(t) = h((t+1)/2) / 2
t = range(-1, 1, length=fine_scale_size+1)[begin:end-1]
Δt = Float64(t.step)
# f(t) = (1 - t^2)*(3/2 + sin(2pi*t))/2
#g(t, n) = t^n
g(t, n) = sum(binomial(n, k)*binomial(n+k, k)*((t - 1)/2)^k for k in 0:n) * sqrt((2n+1)/2) # Legendre Polynomials

"""Measurement Basis Functions"""
#g(t, n) = n % 2 == 1 ? cos(n/2*pi*t) : sin((n-1)/2*pi*t)

"""Total Variation"""
TV(x; Δt=Δt) = Δt*sum(abs.(diff(x)))
∇TV(x; Δt=Δt) = Δt*([-sign.(diff(x)); 0] + [0; sign.(diff(x))])
# Entry wise, this is the following
# [-sign(x[begin+1] - x[begin]); (-sign(x[i+1] - x[i]) + sign(x[i] - x[i-1]) for i in eachindex(x)[begin+1:end-1])...; sign(x[end] - x[end-1])]

"""Graph Laplacian"""
laplacian_matrix(n) = Tridiagonal(-ones(n-1), [1;2*ones(n-2);1],-ones(n-1))
GL(x; Δt=Δt) =  x'*laplacian_matrix(length(x))*x/Δt
∇GL(x; Δt=Δt) =  laplacian_matrix(length(x))*x/Δt

λ = 70 # Total variation regularization parameter
"""Loss Function"""
# L(x; Δt=Δt, λ=λ, A=A, y=y) = 0.5 * norm(A*x - y)^2 + λ*TV(x)
# ∇L(x; Δt=Δt, λ=λ, A=A, y=y) = A'*(A*x - y) + λ*∇TV(x)
L(x; Δt=Δt, λ=λ, A=A, y=y) = 0.5 * norm(A*x - y)^2 + λ*GL(x;Δt)
∇L(x; Δt=Δt, λ=λ, A=A, y=y) = A'*(A*x - y) + λ*∇GL(x;Δt)

"""step size"""
make_step_size(; A, y, Δt, λ, n) = 1 / opnorm(Symmetric(A'A)+λ*Δt*Symmetric(laplacian_matrix(n))) # Inverse of Smoothness of L(x)
# 1 / sqrt(opnorm(Symmetric(A*A'))^2 + λ^2*norm(∇TV(x; Δt))^2)


function is_valid_scale(scale; grid=t)
    n_scale = log2(length(grid)-1)
    return scale ≤ n_scale || throw(ArgumentError("scale must be ≤ than the number of scales $n_scale"))
end

function make_measurement_matrix(scale; grid=t, n_moments=n_moments)
    is_valid_scale(scale)
    skip = scale_to_skip(scale)
    t = coarsen(grid, skip)
    n = 1:n_moments
    A = g.(t', n) # A[i, j] = g(t[j], n[i])
    return A
end

function make_problem(; grid=t)
    Δt = grid.step |> Float64
    x = f.(grid) * Δt
    A = make_measurement_matrix(1; grid)
    y = A*x
    return A, x, y
end

function scale_problem(A, y; scale=1)
    is_valid_scale(scale)
    skip = scale_to_skip(scale)
    A = coarsen(A, skip; dims=2)
    y = coarsen(y, skip)
    return A, y
end

function interpolate_solution(x)
    return interpolate(x, 2; degree=1) # twice as many points (minus 1)
end

proj(y; sum_constraint=1) = proj_scaled_simplex(y; S=sum_constraint)
# proj(y; sum_constraint=1) = positive_normalize_sum(y; sum_constraint)
ReLU(x) = max(0, x)

function positive_normalize_sum(x; sum_constraint=1)
    x = abs.(x)
    x .*= sum_constraint/sum(x)
    return x
end

"""
    proj_scaled_simplex(y; S=1)

Projects (in Euclidian distance) the vector y into the scaled simplex:

    {y | y[i] ≥ 0 and sum(y) = S}

[1] Yunmei Chen and Xiaojing Ye, "Projection Onto A Simplex", 2011
"""
function proj_scaled_simplex(y; S=1)
    n = length(y)

    if n==1 # quick exit for trivial length-1 "vectors" (i.e. scalars)
        return [one(eltype(y))]
    end

    y_sorted = sort(y[:]) # Vectorize/extract input and sort all entries
    i = n - 1
    t = 0 # need to ensure t has scope outside the while loop
    while true
        t = (sum(@view y_sorted[i+1:end]) - S) / (n-i)
        if t >= y_sorted[i]
            break
        else
            i -= 1
        end

        if i >= 1
            continue
        else # i == 0
            t = (sum(y_sorted) - S) / n
            break
        end
    end
    return ReLU.(y .- t)
end

relative_error(a, b) = norm(a - b) / norm(b)

function initialize_x(size)
    # x = ones(size)
    x = abs.(randn(size))
    x ./= sum(x)
    return x
end

function solve_problem(A, y; x_init=initialize_x(size(A, 2)), grad_tol=0.0001, rel_tol=0.002, max_itr=1500, Δt=Δt)
    n = length(x_init)
    sum_constraint = n / fine_scale_size

    sum(proj(x_init; sum_constraint)) ≈ sum_constraint || throw(ArgumentError("x_init does not sum to $sum_constraint"))

    x = x_init
    α = make_step_size(; A, y, Δt, λ, n)
    i = 1
    norm_grad_init = norm(∇L(x; Δt, A, y))
    grad = ∇L(x; Δt, A, y)

    while norm(grad)/norm_grad_init > grad_tol #relative_error(A*x, y) > rel_tol
        x = proj(x - α*grad; sum_constraint)
        grad = ∇L(x; Δt, A, y)

        i += 1
        if i > max_itr
            @warn "Reached maximum number of iterations $max_itr"
            break
        end
    end
    return x, i
end

A, x, y = make_problem()

xhat, n_iterations = solve_problem(A, y)


# p = plot()
# for n in 1:n_moments
#     plot!(t, g.(t, n))
# end
# display(p)

p = plot(t, x)
plot!(t, xhat)
display(p)
@show sum(x)
@show sum(xhat)
@show L(xhat)
@show norm(∇L(xhat))

#N = 50
#plot(t, sum(solve_problem(A, y)[1] for _ in 1:N) / N)
