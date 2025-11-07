"""
Script for comparing multiscale to single scale solving on the following problem for a given
vector y and matrix A.

minₓ 0.5‖Ax - y‖₂² + λGL(x) s.t. ‖x‖₁ = 1, and x ≥ 0,

where GL is the Graph Laplacian regularizer 0.5x'Gx. This is the discrete version of the
following continuous problem,

min_f 0.5‖A(f) - y‖₂² + 0.5λ‖f′‖₂² ‖ s.t. ‖f‖₁ = 1, and f(t) ≥ 0,

where x[i] = f(t[i])Δt
"""

using Random
using LinearAlgebra
using Plots
using Statistics
using BenchmarkTools

########################
# Function Definitions #
########################

### Helpers ###

"""Converts the scale number s to the number of points to skip over in a coarsening of the problem"""
scale_to_skip(s) = 2^(s-1)

"""Converts the number of points in a discretization to the maximum scale number S"""
points_to_scale(n) = Int(log2(n-1))

"""Efficient calculation of ‖x‖₂²"""
norm2(x) = sum(x -> x^2, x)

"""Rectified Linear Unit"""
ReLU(x) = max(0, x)

"""Linearly interpolates a vector x.
If J=length(x), interpolate(x) will have length 2J-1."""
function interpolate(x)
    I = 2length(x)-1
    x̲ = zeros(I)
    for i in 1:I
        if iseven(i)
            x̲[i] = 0.5*(x[i÷2] + x[i÷2 + 1])
        else # i is odd
            x̲[i] = x[(i+1) ÷ 2]
        end
    end
    return x̲
end

### Regularizer ###

"""Graph Laplacian Matrix"""
laplacian_matrix(n) = SymTridiagonal([1;2*ones(n-2);1],-ones(n-1))

"""
    GL(x; Δt, scale=1)

Compute cx'Gx efficiently where
c = 0.5 / Δt^3 / scale_to_skip(scale)
G = laplacian_matrix(length(x))

Equivalent to c*norm(diff(x))^2.
"""
function GL(x; Δt, scale=1)
    n = length(x)
    total = (x[1] - x[2])^2 # type stable and saves a single call to initialize total
    for i in 2:(n-1)
        total += (x[i] - x[i+1])^2
    end
    return 0.5*total / Δt^3 / scale_to_skip(scale)
end

"""Gradient of GL(x)"""
∇GL(x; Δt, scale=1) =  laplacian_matrix(length(x)) * x / (Δt^3 * scale_to_skip(scale))

"""Efficient implementation of ∇GL(x) that stores the result in z"""
function ∇GL!(z, x; Δt, scale=1)
    n=length(x)
    z[1] = x[1] - x[2]
    for i in 2:(n-1)
        z[i] = -x[i-1] + 2x[i] - x[i+1]
    end
    z[n] = x[n] - x[n-1]
    Δt3_scaled = 1 / Δt^3 / scale_to_skip(scale)
    z .*= Δt3_scaled
    return z
end

### Problem Creation ###

"""Legendre Polynomial Measurement Basis Functions"""
g(t, m) = sum(binomial(m, k)*binomial(m+k, k)*((t - 1)/2)^k for k in 0:m) * sqrt((2m+1)/2)

"""Samples the Legendre polynomials on t"""
function make_measurement_matrix(t; n_measurements)
    m = 1:n_measurements
    A = g.(t', m) # equivalent to A[i, j] = g(t[j], m[i]) for all (i, j)
    # End points should be half as big to follow trapezoid rule
    # when multiplying with x
    A[:, begin] ./= 2
    A[:, end] ./= 2
    return A
end

"""
Makes A, x, and y where y is a (possibly noisy) version of A*x where

x[i] = f(t[i]) * Δt
"""
function make_problem(; t, f, σ=0, n_measurements)
    Δt = t.step |> Float64
    x = @. f(t) * Δt
    A = make_measurement_matrix(t; n_measurements)
    ϵ = randn(size(A, 1))
    y_clean = A*x
    y = y_clean + σ*ϵ/norm(ϵ)
    return A, x, y
end

### Problem Solving Functions ###

"""
    proj_scaled_simplex(y; S=1)

Projects (in Euclidian distance) the vector y into the scaled simplex:

    {y | y[i] ≥ 0 and sum(y) = sum_constraint}

[1] Yunmei Chen and Xiaojing Ye, "Projection Onto A Simplex", 2011
"""
function proj!(y; sum_constraint=1)
    n = length(y)

    y_sorted = sort(y) # Vectorize/extract input and sort all entries, will make a copy
    total = y_sorted[n]
    i = n - 1
    t = 0 # need to ensure t has scope outside the while loop
    while true
        t = (total - sum_constraint) / (n-i)
        y_i = y_sorted[i]
        if t >= y_i
            break
        else
            total += y_i
            i -= 1
        end

        if i >= 1
            continue
        else # i == 0
            t = (total - sum_constraint) / n
            break
        end
    end
    @. y = ReLU(y - t)
end

"""
Initializes x to a random point in the set
{x | ‖x‖₁ = sum_constraint, and x ≥ 0}.
Note this is not a *uniformly* random point on this set!
"""
function initialize_x(size; sum_constraint=1)
    x = rand(size) # uniform entries between [0, 1]. Ensures positive values unlike randn
    normalization = sum_constraint / sum(x)
    x .*= normalization
    return x
end

"""
    make_step_size(; A, Δt, λ, scale=1)

Calculates the inverse approximate smoothness of the loss function to use as the step size.

The exact smoothness is
opnorm(A'*A + (λ/Δt^3/scale_to_skip(scale))*laplacian_matrix(size(A, 2)))
which is upper bounded by
opnorm(A'*A) + (λ/Δt^3/scale_to_skip(scale)) * opnorm(laplacian_matrix(size(A, 2)))
with triangle inequality. We rewrite opnorm(A'*A) == opnorm(A*A'), because A*A' is a much smaller m×m matrix that is faster to compute and take its operator norm. And the Laplacian Matrix's opnorm is upper bounded by 4.
"""
make_step_size(; A, Δt, λ, scale=1) = 1 / (opnorm(Symmetric(A*A')) + 4λ/Δt^3/scale_to_skip(scale))

"""
    solve_problem(A, y;
        L, ∇L!, Δt, λ=λ, sum_constraint=1, scale=1, n=(size(A, 2) - 1) ÷ scale_to_skip(scale) + 1, x_init=initialize_x(n; sum_constraint), loss_tol=0.01, max_itr=8000, ignore_warnings=false)

Main algorithm for solving minₓ L(x) at a single scale.
"""
function solve_problem(A, y; L, ∇L!, Δt, λ, sum_constraint=1, scale=1, n=(size(A, 2) - 1) ÷ scale_to_skip(scale) + 1, x_init=initialize_x(n; sum_constraint), loss_tol=0.01, max_itr=8000, ignore_warnings=false)

    @assert n == length(x_init)

    g = zeros(n) # gradient

    if scale != 1 # Avoid making a copy when scale == skip == 1
        skip = scale_to_skip(scale)
        A = A[:, begin:skip:end] * skip
    end

    α = make_step_size(; A, Δt, λ, scale)

    x = x_init # relabel
    i = 1 # iteration counter

    ∇L!(g, x; Δt, A, y, λ, scale) # obtain the first gradient, save it in g

    loss_per_itr = Float64[] # record the loss at each iteration
    push!(loss_per_itr, L(x; Δt, A, y, λ, scale))

    while loss_per_itr[i] > loss_tol
        @. x -= α * g
        proj!(x; sum_constraint)
        ∇L!(g, x; Δt, A, y, λ,scale) # next iteration's gradient (so it can be used in while loop condition)
        if i == max_itr
            ignore_warnings || @warn "Reached maximum number of iterations $max_itr"
            break
        end

        i += 1

        push!(loss_per_itr, L(x; Δt, A, y, λ, scale))
    end

    return x, i, loss_per_itr
end

"""
    solve_problem_multiscale(A, y;
        L, ∇L!, Δt, λ, sum_constraint=1,  n_scales=points_to_scale(size(A, 2)),x_init=initialize_x(3; sum_constraint= sum_constraint / scale_to_skip(n_scales)), loss_tol=0.01, max_itr=8000, ignore_warnings=false, show_plot=false)

Main algorithm for solving minₓ L(x) at multiple scales.
"""
function solve_problem_multiscale(A, y; L, ∇L!, Δt, λ, sum_constraint=1,  n_scales=points_to_scale(size(A, 2)),x_init=initialize_x(3; sum_constraint= sum_constraint / scale_to_skip(n_scales)), loss_tol=0.01, max_itr=8000, ignore_warnings=false, show_plot=false)

    all_iterations = zeros(Int, n_scales)

    x_S, i_S, _ = solve_problem(A, y; L, ∇L!, λ, sum_constraint = sum_constraint / scale_to_skip(n_scales),
        x_init, ignore_warnings=true, max_itr=1, loss_tol=0, Δt, scale=n_scales) # force one gradient step

    if show_plot; p = plot(range(-1,1,length=length(x_S)), x_S); end

    x_s = interpolate(x_S)

    all_iterations[n_scales] = i_S

    # Middle scale solves
    for scale in (n_scales-1):-1:2 # Count down from larger to smaller scales
        x_s, i_s,_ = solve_problem(A, y; L, ∇L!, λ, sum_constraint = sum_constraint / scale_to_skip(scale),
            x_init=x_s, ignore_warnings=true, max_itr=1, loss_tol=0, Δt, scale) # force one gradient step

        if show_plot; plot!(range(-1,1,length=length(x_s)), x_s); end
        x_s = interpolate(x_s)

        all_iterations[scale] = i_s
    end

    # Finest scale solve
    x_1, i_1, loss_per_itr = solve_problem(A, y; L, ∇L!, Δt, λ, sum_constraint, x_init=x_s, max_itr, loss_tol, scale=1, ignore_warnings)


    if show_plot; plot!(range(-1,1,length=length(x_1)), x_1); display(p); end

    all_iterations[1] = i_1

    return x_1, all_iterations, loss_per_itr
end

#######################
# Start of Benchmarks #
#######################

Random.seed!(3141592653)

n_measurements = 5 # Number of Legendre polynomial measurements
scales = 3:12 # What maximum scales S to run the benchmark
λ = 1e-4 # Graph Laplacian regularization parameter
σ = 0.05 # Percent Gaussian noise in measurement y
percent_loss_tol = 0.05 # Iterate until the loss is within 5% of the ground truth

# Ground truth function
f(t) = -2.625t^4 - 1.35t^3 + 2.4t^2 + 1.35t + 0.225

loss_tol_per_scale = zeros(length(scales)) # record loss tolerance used at each scale
# these value should stabilize as the number of points grow

# need to extend the default amount of time to run single scale benchmarks
seconds_single_benchmarks = [5.0 for _ in scales]
seconds_single_benchmarks[scales .== 12] .= 85.0
seconds_single_benchmarks[scales .== 11] .= 25.0
seconds_single_benchmarks[scales .== 10] .= 10.0

# Initialize the benchmark suite
suite = BenchmarkGroup()

# Loop over each scale and create the benchmarks
for (s, (n_scales, seconds_single_benchmark)) in enumerate(zip(scales, seconds_single_benchmarks))

    fine_scale_size = 2^n_scales + 1
    t = range(-1, 1, length=fine_scale_size)
    Δt = Float64(t.step)

    A, x, y = make_problem(; t, f, σ, n_measurements)

    """Loss Function"""
    L(x; Δt=Δt, λ=λ, A=A, y=y, scale=1) = 0.5 * norm2(A*x .- y) + λ .* GL(x; Δt, scale)

    """Gradient of Loss function L(x)"""
    ∇L(x; Δt, λ, A, y, scale=1) = A'*(A*x .- y) .+ λ .* ∇GL(x; Δt, scale)

    """Efficient in-place version of ∇L(x), storing the result in z."""
    function ∇L!(z, x; Δt, λ, A, y, scale=1)
        ∇GL!(z, x; Δt, scale)
        mul!(z, A', A*x .- y, 1, λ) # mul!(C, A, B, α, β) == ABα+Cβ
        # This mul! function call is equivalent to
        # z .= A' * (A * x .- y) .* 1 .+ z .* λ
    end

    # compile
    loss_tol = L(x)*(1 + percent_loss_tol) # want xhat to be at least as good as our true x
    loss_tol_per_scale[s] = loss_tol
    xhat, n_itr_single, loss_per_itr_single = solve_problem(A, y; L, ∇L!, loss_tol, Δt, λ)
    xhat_multi, n_itr_multi, loss_per_itr_multi = solve_problem_multiscale(A, y; L, ∇L!, loss_tol, Δt, λ, show_plot=true)

    # Plot a typical solution
    p = plot(; xlabel="t at scale $n_scales", ylabel="density")
    plot!(t, x; label="true distribution")
    plot!(t, xhat;label="single scale")
    plot!(t, xhat_multi; label="multi-scale")
    display(p)

    # Plot the loss curves at the finest scale, ignoring the initialization
    p = plot(;xlabel="# iterations at scale $n_scales", ylabel="loss", yaxis=:log10)
    plot!(loss_per_itr_single[begin+1:end]; label="single scale")
    plot!(loss_per_itr_multi[begin+1:end]; label="multi scale")
    display(p)

    # Create the benchmark
    suite["single"][n_scales] = @benchmarkable solve_problem($A, $y; L=$L, ∇L! = $∇L!, loss_tol=$loss_tol, Δt=$Δt, λ=$λ) seconds=seconds_single_benchmark samples=100

    suite["multi"][n_scales] = @benchmarkable solve_problem_multiscale($A, $y; L=$L, ∇L! = $∇L!, loss_tol=$loss_tol, Δt=$Δt, λ=$λ) seconds=1.0 samples=100

end

display(suite)

results = run(suite, verbose=true)

# How to extract
# bmk_single = results["single"][scale_s]
# median(bmk_single).time # in ns
# median(bmk_single).memory # in bytes (divide by 2^20 for MiB or 10^6 for MB)
# bmk_single.params.samples # number of times the benchmark was run

################
# Plot Results #
################

top_quantile = 0.95
bot_quantile = 0.05

function get_time(f, benchmark)
    try # See if BenchmarkTools already calculated f, and can look up the stat
        return [f(results[benchmark][S]).time * 1e-6 for S in scales] # in ms
    catch err
        if typeof(err) == MethodError
            return [f(results[benchmark][S].times) * 1e-6 for S in scales]
        else
            throw(err)
        end
    end
end

single_median_times = get_time(median, "single")
multi_median_times = get_time(median, "multi")

single_top_times = get_time(x -> quantile(x, top_quantile), "single")
single_bot_times = get_time(x -> quantile(x, bot_quantile), "single")
multi_top_times = get_time(x -> quantile(x, top_quantile), "multi")
multi_bot_times = get_time(x -> quantile(x, bot_quantile), "multi")

problem_sizes = @. 2^scales + 1

p = plot(;
    xlabel="problem size (number of points)",
    ylabel="median time (ms)",
    xticks=(problem_sizes .- 1),
    yticks=[10. ^n for n in -2:3],
    xaxis=:log2,
    yaxis=:log10,
    size=(450,250),
    )
plot!(problem_sizes, single_median_times;
    ribbon=(single_median_times - single_bot_times, single_top_times - single_median_times),
    fillalpha=0.2,
    label="single scale",
    marker=true,
    markerlinestyle=:white)
plot!(problem_sizes, multi_median_times;
    ribbon=(multi_median_times - multi_bot_times, multi_top_times - multi_median_times),
    fillalpha=0.2,
    label="multi scale",
    marker=true)
plot!(;legend=:topleft)
display(p)
