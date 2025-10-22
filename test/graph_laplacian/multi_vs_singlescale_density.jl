"""
Script for comparing multiscale to single scale solving on the following problem for a given vector y and matrix A.

minₓ 0.5||Ax - y||₂² + λGL(x) s.t. ||x||₁ = 1, and x ≥ 0

where GL is the Graph Laplacian regularizer
"""

using BlockTensorFactorization
using Random
using LinearAlgebra
using Plots

n_scales = 8
n_measurements = 8
fine_scale_size = 2^n_scales + 1
scale_to_skip(s) = 2^(s-1)
n_points_to_n_scales(n) = Int(log2(n-1))
# t = range(0, 1, length=fine_scale_size)
h(t) = -84t^4 + 146.4t^3 - 74.4t^2 + 12t
f(t) = h((t+1)/2) / 2
t = range(-1, 1, length=fine_scale_size+1)[begin:end-1]
Δt = Float64(t.step)
# f(t) = (1 - t^2)*(3/2 + sin(2pi*t))/2

"""Measurement Basis Functions"""
#g(t, n) = n % 2 == 1 ? cos(n/2*pi*t) : sin((n-1)/2*pi*t)
#g(t, n) = t^n
g(t, n) = sum(binomial(n, k)*binomial(n+k, k)*((t - 1)/2)^k for k in 0:n) * sqrt((2n+1)/2) # Legendre Polynomials
g.(t, 0)

"""Graph Laplacian"""
laplacian_matrix(n) = Tridiagonal(-ones(n-1), [1;2*ones(n-2);1],-ones(n-1))
GL(x; Δt=Δt) =  x'*laplacian_matrix(length(x))*x/Δt^2
∇GL(x; Δt=Δt) =  laplacian_matrix(length(x))*x/Δt^2

λ = 0.1 # Total variation regularization parameter
"""Loss Function"""
L(x; Δt=Δt, λ=λ, A=A, y=y) = 0.5 * norm(A*x - y)^2 + λ*GL(x;Δt)
∇L(x; Δt=Δt, λ=λ, A=A, y=y) = A'*(A*x - y) + λ*∇GL(x;Δt)

"""step size"""
make_step_size(; A, y, Δt, λ, n) = 1 / opnorm(Symmetric(A'*A)+(λ/Δt^2)*Symmetric(laplacian_matrix(n))) # Inverse of Smoothness of L(x)
# 1 / sqrt(opnorm(Symmetric(A*A'))^2 + λ^2*norm(∇TV(x; Δt))^2)

function is_valid_scale(scale; grid=t)
    n_scale = log2(length(grid)-1)
    return scale ≤ n_scale || throw(ArgumentError("scale must be ≤ than the number of scales $n_scale"))
end

function make_measurement_matrix(scale; grid=t, n_measurements=n_measurements)
    is_valid_scale(scale)
    skip = scale_to_skip(scale)
    t = coarsen(grid, skip)
    n = 0:n_measurements
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
    y = y / scale_to_skip(scale)
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
    #x = ones(size)
    x = abs.(randn(size))
    x ./= sum(x)
    return x
end

function solve_problem(A, y; x_init=initialize_x(size(A, 2)), loss_tol=L(x), grad_tol=0.0001, rel_tol=0.002, max_itr=1500, Δt=Δt, λ=λ, ignore_warnings=false)
    n = length(x_init)
    sum_constraint = n / fine_scale_size
    sum(proj(x_init; sum_constraint)) ≈ sum_constraint || throw(ArgumentError("x_init does not sum to $sum_constraint"))

    x = x_init
    α = make_step_size(; A, y, Δt, λ, n)
    i = 1
    norm_grad_init = norm(∇L(x; Δt, A, y, λ))
    grad = ∇L(x; Δt, A, y)

    while L(x; Δt, A, y, λ) > loss_tol #norm(grad)/norm_grad_init > grad_tol #relative_error(A*x, y) > rel_tol
        x = proj(x - α*grad; sum_constraint)
        grad = ∇L(x; Δt, A, y, λ)

        i += 1
        if i > max_itr
            ignore_warnings || @warn "Reached maximum number of iterations $max_itr"
            break
        end
    end
    return x, i
end

function solve_problem_multiscale(A, y; x_init=initialize_x(3), loss_tol=L(x), grad_tol=0.0001, rel_tol=0.002, max_itr=1500, Δt=Δt, ignore_warnings=false, n_scales=n_points_to_n_scales(size(A, 2)))

    # Coarsest scale solve
    A_S, y_S = scale_problem(A, y; scale=n_scales)

    x_S, _ = solve_problem(A_S, y_S;
        x_init, ignore_warnings=true, max_itr=1, grad_tol=0, Δt=Δt * scale_to_skip(n_scales)) # force one gradient step
    # p = plot(coarsen(t, scale_to_skip(n_scales)), x_S)
    x_s = interpolate_solution(x_S)
    # Middle scale solves
    for scale in (n_scales-1):-1:2 # Count down from larger to smaller scales
        A_s, y_s = scale_problem(A, y; scale)
        x_s, _ = solve_problem(A_s, y_s;
            x_init=x_s, ignore_warnings=true, max_itr=1, Δt=Δt * scale_to_skip(scale)) # force one gradient step
        # p = plot!(coarsen(t, scale_to_skip(scale)), x_s)
        x_s = interpolate_solution(x_s)
    end

    # Finest scale solve
    x_1, n_iterations = solve_problem(A, y; x_init=x_s, max_itr, loss_tol)
    # p = plot!(t, x_1)
    # display(p)
    return x_1, n_iterations
end

A, x, y = make_problem()

@time xhat, n_iterations = solve_problem(A, y)

@time xhat_multi, n_iterations_multi = solve_problem_multiscale(A, y)

# p = plot()
# for n in 1:n_measurements
#     plot!(t, g.(t, n))
# end
# display(p)

p = plot(t, x)
plot!(t, xhat)
plot!(t, xhat_multi)
display(p)
@show sum(x)
@show sum(xhat)
@show L(x)
@show L(xhat)
@show norm(∇L(xhat))
println()
@show sum(xhat_multi)
@show L(xhat_multi)
@show norm(∇L(xhat_multi))

benchmark = true
using BenchmarkTools
if benchmark
bmk_single = @benchmark solve_problem(A, y)
display(bmk_single)

bmk_multi = @benchmark solve_problem_multiscale(A, y)
display(bmk_multi)
end

scale = n_scales - 3
y_s = coarsen(A, scale_to_skip(scale);dims=2) * coarsen(x, scale_to_skip(scale))
