"""
Script for comparing multiscale to single scale solving on the following problem for a given vector y and matrix A.

minₓ 0.5||Ax - y||₂² + λGL(x) s.t. ||x||₁ = 1, and x ≥ 0

where GL is the Graph Laplacian regularizer
"""

using BlockTensorFactorization
using Random
using LinearAlgebra
using Plots
using Statistics
using BenchmarkTools

n_measurements = 5

h(t) = -84t^4 + 146.4t^3 - 74.4t^2 + 12t
f(t) = h((t+1)/2) / 2

λ = 0.1 # Total variation regularization parameter
σ = 0.01 # percent Gaussian noise in measurement y
percent_loss_tol = 0.01 # iterate until the loss is within 1% of the optimal loss

scale_to_skip(s) = 2^(s-1)
n_points_to_n_scales(n) = Int(log2(n-1))

"""Measurement Basis Functions"""
#g(t, n) = n % 2 == 1 ? cos(n/2*pi*t) : sin((n-1)/2*pi*t)
#g(t, n) = t^n
g(t, n) = sum(binomial(n, k)*binomial(n+k, k)*((t - 1)/2)^k for k in 0:n) * sqrt((2n+1)/2) # Legendre Polynomials

"""Graph Laplacian"""
laplacian_matrix(n) = Tridiagonal(-ones(n-1), [1;2*ones(n-2);1],-ones(n-1))

# Although the following implementations are clean, they are slow because of generating the laplacian matrix
# GL_old(x; Δt) =  x'*laplacian_matrix(length(x))*x/Δt^2
# ∇GL_old(x; Δt) =  laplacian_matrix(length(x))*x/Δt^2

GL(x; Δt) =  norm2(diff(x))/Δt^2
function ∇GL(x; Δt)
    dt2 = Δt^2
    out = zero(x)
    n = length(x)
    for i in eachindex(x)
        if i == 1
            out[i] = (x[i] - x[i+1]) / dt2
        elseif i == n
            out[i] = (x[i] - x[i-1]) / dt2
        else
            out[i] = (-x[i-1] + 2x[i] - x[i+1]) / dt2
        end
    end
    return out
end

"""step size"""
# Inverse of Smoothness of L(x)
make_step_size(; A, y, Δt, λ, n) = 1 / (opnorm(Symmetric(A*A'))+4λ/Δt^2)
# Using the fact that the opnorm satisfied triangle inequality,
# We get an upper bound by splitting up the opnorm for the two matrices
# The bound is fairly tight since both matrices are close to a constant times the identity
# The laplacian matrix's opnorm is upper bounded by 4 so we cna skip that computation
# And since A*A' and A'*A have the same opnorm, we use the former since its smaller
# 1 / opnorm(Symmetric(A'*A+(λ/Δt^2)laplacian_matrix(n)))
# 1 / sqrt(opnorm(Symmetric(A*A'))^2 + λ^2*norm(∇TV(x; Δt))^2)

function is_valid_scale(scale; grid)
    n_scale = log2(length(grid)-1)
    return scale ≤ n_scale || throw(ArgumentError("scale must be ≤ than the number of scales $n_scale"))
end

function make_measurement_matrix(scale; grid, n_measurements=n_measurements)
    is_valid_scale(scale; grid)
    skip = scale_to_skip(scale)
    t = coarsen(grid, skip)
    n = 0:n_measurements
    A = g.(t', n) # A[i, j] = g(t[j], n[i])
    return A
end

function make_problem(; grid=t, σ=0)
    Δt = grid.step |> Float64
    x = f.(grid) * Δt
    A = make_measurement_matrix(1; grid)
    ϵ = randn(size(A, 1))
    y_clean = A*x
    y = y_clean + norm(y_clean)*σ*ϵ
    return A, x, y
end

function scale_problem(A, y; grid, scale=1)
    is_valid_scale(scale; grid)
    skip = scale_to_skip(scale)
    A = coarsen(A, skip; dims=2)
    y = y / scale_to_skip(scale)
    return A, y
end

function interpolate_solution(x)
    return interpolate(x, 2; degree=1) # twice as many points (minus 1)
end

function positive_normalize_sum(x; sum_constraint=1)
    x = abs.(x)
    x .*= sum_constraint/sum(x)
    return x
end

ReLU(x) = max(0, x)

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

#proj(y; sum_constraint=1) = proj_scaled_simplex(y; S=sum_constraint)
proj(y; sum_constraint=1) = positive_normalize_sum(y; sum_constraint)

relative_error(a, b) = norm(a - b) / norm(b)

function initialize_x(size)
    #x = ones(size)
    x = abs.(randn(size))
    x ./= sum(x)
    return x
end

function solve_problem(A, y; L, ∇L, Δt, t, x_init=initialize_x(size(A, 2)), loss_tol=0.01, grad_tol=0.0001, rel_tol=0.002, max_itr=4e3,λ=λ, ignore_warnings=false)
    n = length(x_init)
    sum_constraint = n / length(t)
    sum(proj(x_init; sum_constraint)) ≈ sum_constraint || throw(ArgumentError("x_init does not sum to $sum_constraint"))

    x = x_init
    α = make_step_size(; A, y, Δt, λ, n)
    i = 1
    norm_grad_init = norm(∇L(x; Δt, A, y, λ))
    grad = ∇L(x; Δt, A, y)

    loss_tol *= sqrt(n) # scale by problem size

    while norm(grad) > loss_tol #L(x; Δt, A, y, λ) > loss_tol #norm(grad)/norm_grad_init > grad_tol #relative_error(A*x, y) > rel_tol
        x = proj(x .- α .* grad; sum_constraint)
        grad = ∇L(x; Δt, A, y, λ)

        i += 1
        if i > max_itr
            ignore_warnings || @warn "Reached maximum number of iterations $max_itr"
            break
        end
    end
    return x, i
end

function solve_problem_multiscale(A, y; L, ∇L, Δt, t, λ, x_init=initialize_x(3), loss_tol=0.01, grad_tol=0.0001, rel_tol=0.002, max_itr=4e3, ignore_warnings=false, n_scales=n_points_to_n_scales(size(A, 2)))

    # Coarsest scale solve
    A_S, y_S = scale_problem(A, y; grid=t, scale=n_scales)

    x_S, _ = solve_problem(A_S, y_S; L, ∇L, t, λ,
        x_init, ignore_warnings=true, max_itr=1, grad_tol=0, Δt=Δt * scale_to_skip(n_scales)) # force one gradient step
    # p = plot(coarsen(t, scale_to_skip(n_scales)), x_S)
    x_s = interpolate_solution(x_S)
    # Middle scale solves
    for scale in (n_scales-1):-1:2 # Count down from larger to smaller scales
        A_s, y_s = scale_problem(A, y; grid=t, scale)
        x_s, _ = solve_problem(A_s, y_s; L, ∇L, t, λ,
            x_init=x_s, ignore_warnings=true, max_itr=1, Δt=Δt * scale_to_skip(scale)) # force one gradient step
        # p = plot!(coarsen(t, scale_to_skip(scale)), x_s)
        x_s = interpolate_solution(x_s)
    end

    # Finest scale solve
    x_1, n_iterations = solve_problem(A, y; L, ∇L, t, Δt, λ, x_init=x_s, max_itr, loss_tol)
    # p = plot!(t, x_1)
    # display(p)
    return x_1, n_iterations
end

using Profile

profile = true
if profile
    n_scales = 11

    fine_scale_size = 2^n_scales + 1
    t = range(-1, 1, length=fine_scale_size+1)[begin:end-1]
    Δt = Float64(t.step)

    A, x, y = make_problem(; grid=t, σ)

    """Loss Function"""
    L(x; Δt, λ=λ, A=A, y=y) = 0.5 * norm(A*x .- y)^2 .+ λ .* GL(x;Δt)
    ∇L(x; Δt, λ=λ, A=A, y=y) = A'*(A*x .- y) .+ λ .* ∇GL(x;Δt)

    loss_tol = 0.015 # L(x; Δt) * (1 + percent_loss_tol)

    # Compile
    xhat, _ = solve_problem(A, y; L, ∇L, loss_tol, Δt, t, λ)
    xhat_multi, _ = solve_problem_multiscale(A, y; L, ∇L, loss_tol, Δt, t, λ)

    # Plot a typical solution
    p = plot(; xlabel="t at scale $n_scales", ylabel="density")
    plot!(t, x; label="true distribution")
    plot!(t, xhat;label="single scale")
    plot!(t, xhat_multi; label="multi-scale")
    display(p)

    # profile

    #@profview solve_problem(A, y; L, ∇L, loss_tol, Δt, t, λ)
    #@profview solve_problem_multiscale(A, y; L, ∇L, loss_tol, Δt, t, λ)
end

norm(∇L(xhat; Δt))

norm(∇L(xhat_multi; Δt))
L(xhat; Δt)
L(xhat_multi; Δt)
L(x; Δt)
@profview solve_problem(A, y; L, ∇L, loss_tol, Δt, t, λ)
# @profview solve_problem_multiscale(A, y; L, ∇L, loss_tol, Δt, t, λ)

@benchmark solve_problem(A, y; L, ∇L, loss_tol, Δt, t, λ)

@benchmark solve_problem_multiscale(A, y; L, ∇L, loss_tol, Δt, t, λ)

#######################
# Start of Benchmarks #
#######################

scales = 3:12 # 3:12
regularization = []

suite = BenchmarkGroup()

for n_scales in scales

    fine_scale_size = 2^n_scales + 1
    t = range(-1, 1, length=fine_scale_size+1)[begin:end-1]
    Δt = Float64(t.step)

    A, x, y = make_problem(; grid=t, σ)

    """Loss Function"""
    L(x; Δt, λ=λ, A=A, y=y) = 0.5 * norm(A*x .- y)^2 .+ λ .* GL(x;Δt)
    ∇L(x; Δt, λ=λ, A=A, y=y) = A'*(A*x .- y) .+ λ .* ∇GL(x;Δt)

    loss_tol = L(x; Δt) * (1 + percent_loss_tol)

    # Compile
    xhat, _ = solve_problem(A, y; L, ∇L, loss_tol, Δt, t, λ)
    xhat_multi, _ = solve_problem_multiscale(A, y; L, ∇L, loss_tol, Δt, t, λ)

    # Plot a typical solution
    p = plot(; xlabel="t at scale $n_scales", ylabel="density")
    plot!(t, x; label="true distribution")
    plot!(t, xhat;label="single scale")
    plot!(t, xhat_multi; label="multi-scale")
    display(p)

    # Prep the benchmarks
    suite["single"][n_scales] = @benchmarkable solve_problem($A, $y; L=$L, ∇L=$∇L, loss_tol=$loss_tol, Δt=$Δt, t=$t, λ=$λ)
    suite["multi"][n_scales] = @benchmarkable solve_problem_multiscale($A, $y; L=$L, ∇L=$∇L, loss_tol=$loss_tol, Δt=$Δt, t=$t, λ=$λ)

end

run_benchmarks = true

if run_benchmarks

# how to extract
# median(bmk_single).time # in ns
# median(bmk_single).memory # in bytes (divide by 2^20 for MiB or 10^6 for MB)
# bmk_single.params.samples # number of times the benchmark was run

tune!(suite) # tunes the parameters of all the benchmarks

results = run(suite, verbose=true)
end
################
# Plot Results #
################

plot_results = true
if run_benchmarks && plot_results
single_median_times = [median(results["single"][S]).time * 1e-6 for S in scales] # in ns
multi_median_times = [median(results["multi"][S]).time * 1e-6 for S in scales] # in ms

single_mean_times = [mean(results["single"][S]).time * 1e-6 for S in scales] # in ms
multi_mean_times = [mean(results["multi"][S]).time * 1e-6 for S in scales] # in ms

#single_median_times .*= 1e-6 # in ms
#multi_median_times .*= 1e-6 # in ms

problem_sizes = @. 2^scales + 1

p = plot(;
    xlabel="problem size (number of points)",
    ylabel="median time (ms)",
    xticks=(problem_sizes .- 1),
    xaxis=:log2,
    marker=:circle,
    markersize=5,
    yaxis=:log10,
    )
scatter!(problem_sizes, single_median_times; label="single scale")
scatter!(problem_sizes, multi_median_times; label="multi-scale")
plot!(;legend=:topleft)
display(p)

end
