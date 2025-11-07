"""
Script for comparing multiscale to single scale solving on the following problem for a given vector y and matrix A.

minₓ 0.5||Ax - y||₂² + λGL(x) s.t. ||x||₁ = 1, and x ≥ 0

where GL is the Graph Laplacian regularizer
"""

# using BlockTensorFactorization
using Random
using LinearAlgebra
using Plots
using Statistics
using BenchmarkTools

n_measurements = 5

# h(t) = -84t^4 + 146.4t^3 - 74.4t^2 + 12t
f(t) = -2.625t^4 - 1.35t^3 + 2.4t^2 + 1.35t + 0.225 #h((t+1)/2) / 2

λ = 1e-4 #5e-5 #2e-5 #0.0001 # Total variation regularization parameter 0.1, 1e-4, 2e-5
σ = 0.01 # percent Gaussian noise in measurement y
percent_loss_tol = 0.05 # iterate until the loss is within 5% of the optimal loss

#0.01% of the optimal loss

scale_to_skip(s) = 2^(s-1)
n_points_to_n_scales(n) = Int(log2(n-1))

"""Measurement Basis Functions"""
#g(t, n) = n % 2 == 1 ? cos(n/2*pi*t) : sin((n-1)/2*pi*t)
#g(t, n) = t^n / factorial(n)
g(t, n) = sum(binomial(n, k)*binomial(n+k, k)*((t - 1)/2)^k for k in 0:n) * sqrt((2n+1)/2) # Legendre Polynomials

"""Graph Laplacian"""
laplacian_matrix(n) = SymTridiagonal([1;2*ones(n-2);1],-ones(n-1))

# Although the following implementations are clean, they are slow because of generating the laplacian matrix
# GL_old(x; Δt) =  0.5*x'*laplacian_matrix(length(x))*x/Δt^2

t_power = 1
∇GL(x; Δt, scale=1) =  laplacian_matrix(length(x))*x /Δt^t_power / scale_to_skip(scale)

# GL_old(x; Δt, scale=1) =  0.5*norm2(diff(x))/Δt^t_power / scale_to_skip(scale)

function GL(x; Δt, scale=1)
    n = length(x)
    total = (x[1] - x[2])^2 # type stable and saves a single call to initialize total
    for i in 2:(n-1)
        total += (x[i] - x[i+1])^2
    end
    return 0.5*total / Δt^t_power / scale_to_skip(scale)
end

function ∇GL!(z, x; Δt, scale=1)
    n=length(x)
    z[1] = x[1] - x[2]
    for i in 2:(n-1)
        z[i] = -x[i-1] + 2x[i] - x[i+1]
    end
    z[n] = x[n] - x[n-1]
    dt3 = 1 / Δt^t_power / scale_to_skip(scale)
    z .*= dt3
    return z
end

"""step size"""
# Inverse of Smoothness of L(x)
make_step_size(; A, y, Δt, λ, n, scale=1) = 1 / (opnorm(Symmetric(A*A')) + 4λ/Δt^(t_power)/scale_to_skip(scale))
# function make_step_size(; A, y, Δt, λ, n, scale=1)
#     H = Symmetric(A'*A) + 4λ/Δt^(t_power)*laplacian_matrix(n)
#     eig_min, eig_max = extrema(eigvals(H))
#     return 2 / (eig_min + eig_max)
# end
top_eigenval(; A, y, Δt, λ, n, scale=1) = eigmax(Symmetric(A'*A)+(λ/Δt^(t_power)/scale_to_skip(scale))*laplacian_matrix(n))
bot_eigenval(; A, y, Δt, λ, n, scale=1) = eigmin(Symmetric(A'*A)+(λ/Δt^(t_power)/scale_to_skip(scale))*laplacian_matrix(n))
#scale_to_skip(scale) / (Δt^(-1) + 4λ*Δt^(-3))
#
# 1 / opnorm(Symmetric(A'*A+(λ/Δt^(t_power)/scale_to_skip(scale))*laplacian_matrix(n)))
#1 / (Δt^(-1) + 4λ*Δt^(-3)/scale_to_skip(scale))
#1 / opnorm(Symmetric(A'*A+(λ/Δt^(t_power)/scale_to_skip(scale))*laplacian_matrix(n)))
#1 / (opnorm(Symmetric(A*A'))+4λ*Δt^(-2))
#1 / (Δt^(-1) + 4λ*Δt^(-3))
#1.45e-6 at scale 12
# make_step_size(; A, y, Δt, λ, n) = 1 / (opnorm(Symmetric(A*A'))+4λ*Δt^(-3))
# Using the fact that the opnorm satisfied triangle inequality,
# We get an upper bound by splitting up the opnorm for the two matrices
# The bound is fairly tight since both matrices are close to a constant times the identity
# The laplacian matrix's opnorm is upper bounded by 4 so we cna skip that computation
# And since A*A' and A'*A have the same opnorm, we use the former since its smaller
# 1 / opnorm(Symmetric(A'*A+(λ/Δt^2)laplacian_matrix(n)))
# 1 / sqrt(opnorm(Symmetric(A*A'))^2 + λ^2*norm(∇TV(x; Δt))^2)
# and opnorm(A*A') should be 1 if A is orthonormal (or 1/Δt if it has not been normalized

"""
    coarsen(Y::AbstractArray, scale::Integer; dims=1:ndims(Y))

Coarsens or downsamples `Y` by `scale`. Only keeps every `scale` entries along the dimensions specified.

Example
=======

Y = randn(12, 12, 12)

coarsen(Y, 2) == Y[begin:2:end, begin:2:end, begin:2:end]

coarsen(Y, 4; dims=(1, 3)) == Y[begin:4:end, :, begin:4:end]

coarsen(Y, 3; dims=2) == Y[:, begin:3:end, :]
"""
coarsen(Y::AbstractArray, scale::Integer; dims=1:ndims(Y), kwargs...) =
    Y[(d in dims ? axis[begin:scale:end] : axis for (d, axis) in enumerate(axes(Y)))...]

# Using axis[begin:scale:end] rather than 1:scale:size(Y, d) for more flexible indexing

"""
    interpolate(Y, scale; dims=1:ndims(Y), degree=0, kwargs...)

Interpolates Y to a larger array with repeated values.

Keywords
========
`scale`. How much to scale up the size of `Y`.
A dimension with size `k` will be scaled to `scale*k - (scale - 1) = scale*(k-1) + 1`

`dims`:`1:ndims(Y)`. Which dimensions to interpolate.

`degree`:`0`. What degree of interpolation to use. `0` is constant interpolation, `1` is linear.

Like the opposite of [`coarsen`](@ref).

Example
=======

julia> Y = collect(reshape(1:6, 2, 3))
2×3 Matrix{Int64}:
 1  3  5
 2  4  6

julia> interpolate(Y, 2)
3×5 Matrix{Int64}:
 1  1  3  3  5
 1  1  3  3  5
 2  2  4  4  6

julia> interpolate(Y, 3; dims=2)
2×7 Matrix{Int64}:
 1  1  1  3  3  3  5
 2  2  2  4  4  4  6

julia> interpolate(Y, 1) == Y
true
"""
function interpolate(Y::AbstractArray, scale; dims=1:ndims(Y), degree=0, kwargs...)
    # Quick exit if no interpolation is needed
    if scale == 1 || isempty(dims)
        return Y
    end

    Y = repeat(Y; inner=(d in dims ? scale : 1 for d in 1:ndims(Y)))

    # Chop the last slice of repeated dimensions since we only interpolate between
    # the values
    chop = (d in dims ? axis[begin:end-scale+1] : axis for (d, axis) in enumerate(axes(Y)))
    Y = Y[chop...]

    if degree == 0
        return Y
    elseif degree == 1 && scale == 2 # TODO generalize linear_smooth to other scales
        return linear_smooth(Y; dims, kwargs...)
    else
        error("interpolation of degree=$degree with scale=$scale not supported (YET!)")
    end
end

norm2(x) = sum(x -> x^2, x)

function is_valid_scale(scale; grid)
    n_scale = log2(length(grid)-1)
    return scale ≤ n_scale || throw(ArgumentError("scale must be ≤ than the number of scales $n_scale"))
end

function make_measurement_matrix(t; n_measurements=n_measurements)
    n = 1:n_measurements
    A = g.(t', n) # A[i, j] = g(t[j], n[i])
    # End points should be half as big to use trapezoid rule
    A[:, begin] ./= 2
    A[:, end] ./= 2
    return A
end

function make_problem(; t, σ=0)
    Δt = t.step |> Float64
    x = f.(t)
    A = make_measurement_matrix(t) .* Δt
    ϵ = randn(size(A, 1))
    y_clean = A*x
    #y = y_clean + norm(y_clean)*σ*ϵ
    y = y_clean + σ*ϵ/norm(ϵ)
    return A, x, y
end

function noise_amount(scale;σ=0)
    A,x,y=make_problem(;t=t[begin:scale_to_skip(scale):end],σ)
    return 0.5norm2(A*x - y)
end

# function scale_problem!(y_out, A, y; grid, scale=1)
#     is_valid_scale(scale; grid)
#     skip = scale_to_skip(scale)
#     A = coarsen(A, skip; dims=2)
#     @. y_out = y / skip
#     return A, y_out
# end

function interpolate_solution(x)
    return interpolate(x, 2; degree=1) # twice as many points (minus 1)
end


function linear_smooth(Y; dims=1:ndims(Y), kwargs...)
    return _linear_smooth!(1.0 * Y, dims)
    # makes a copy of Y and ensures the type can hold float like elements
end

function _linear_smooth!(Y, dims)
    all_dims = 1:ndims(Y)
    for d in dims
        axis = axes(Y, d)
        Y1 = @view Y[(i==d ? axis[begin+1:end-1] : (:) for i in all_dims)...]
        Y2 = @view Y[(i==d ? axis[begin+2:end] : (:) for i in all_dims)...]

        @. Y1 = 0.5 * (Y1 + Y2)
    end
    return Y
end

function positive_normalize_sum!(x; sum_constraint=1)
    @. x = abs(x)
    normalize_amount = sum_constraint / sum(x)
    x .*= normalize_amount
end

ReLU(x) = max(0, x)

"""
    proj_scaled_simplex(y; S=1)

Projects (in Euclidian distance) the vector y into the scaled simplex:

    {y | y[i] ≥ 0 and sum(y) = S}

[1] Yunmei Chen and Xiaojing Ye, "Projection Onto A Simplex", 2011
"""
function proj_scaled_simplex!(y; S=1)
    n = length(y)

    y_sorted = sort(y) # Vectorize/extract input and sort all entries, will make a copy
    total = y_sorted[n]
    i = n - 1
    t = 0 # need to ensure t has scope outside the while loop
    while true
        t = (total - S) / (n-i)
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
            t = (total - S) / n
            break
        end
    end
    @. y = ReLU(y - t)
end

function my_rand(x)
    y = abs.(randn(x))
    ysum = 0.9sum(y)
    y ./= ysum
    return y
end

"""
Euclidean projection onto the set
    {y | y ≥ 0, w'y = S}
assuming w ≥ 0.

[2] Guillaume Perez, Sebastian Ament, Carla Gomes, Michel Barlaud, "Efficient projection algorithms onto the weighted 1 ball". Algorithm 2.
"""
function proj_general_simplex!(y; S=1, w=one(y))
    n = length(y)

    @assert length(y) == length(w) "lengths must match"
    @assert all(w_i -> w_i > 0, w) "w is not positive."

    # Sort according to y ./ w
    z = y ./ w
    J = sortperm(z)

    # Initialize sums
    j = J[n]
    total = y[j] * w[j]
    w_total = w[j]^2

    # Initialize counters
    i = n - 1
    t = 0 # need to ensure t has scope outside the while loop
    while true
        t = (total - S) / w_total
        j = J[i]
        if t >= z[j]
            break
        else
            w_j = w[j]
            total += y[j] * w_j
            w_total += w_j^2
            i -= 1
        end

        if i >= 1
            continue
        else # i == 0
            t = (total - S) / w_total
            break
        end
    end
    @. y = ReLU(y - t * w)
end

#@benchmark proj_general_simplex!(b; w=c) setup=((b,c)=(rand(1000),rand(1000)))

# @benchmark proj_scaled_simplex!(b) setup=(b=rand(10000))
# @benchmark proj_scaled_simplex2!(b) setup=(b=rand(10000))

#@benchmark proj_general_simplex2!(b; w=c) setup=((b,c)=(rand(1000),rand(1000)))

# @benchmark proj_general_simplex2!(b, c, t) setup=((b,c,t)=(my_rand(1000),rand(1000),Float64(1.0)))

function projsplx!(b::Vector{T}, c::Vector{T}, τ::T) where T

    n = length(b)
    bget = false

    @assert length(b) == length(c) "lengths must match"
    @assert minimum(c) > 0 "c is not positive."

    idx = sortperm(b./c, rev=true)
    tsum = csum = zero(T)

    @inbounds for i = 1:n-1
        j = idx[i]
        tsum += b[j]*c[j]
        csum += c[j]^2 #c[j]*c[j]
        tmax = (tsum - τ) / csum
        if tmax >= b[idx[i+1]] / c[idx[i+1]]
            bget = true
            break
        end
    end

    if !bget
        p = idx[n]
        tsum += b[p]*c[p]
        csum += c[p]^2#c[p]*c[p]
        tmax = (tsum - τ) / csum
    end

    for i = 1:n
        @inbounds b[i] = max(b[i] - c[i]*tmax, 0)
    end

    return

end

function proj!(y; sum_constraint=1)
    #positive_normalize_sum!(y; sum_constraint)
    proj_scaled_simplex!(y; S=sum_constraint)
    #w = ones(length(y))
    #w[1] = 0.5; w[end] = 0.5
    #proj_general_simplex!(y; S=sum_constraint, w)
end

function initialize_x(size; sum_constraint=1)
    #x = ones(size)
    x = rand(size) #.+ 1
    #x = abs.(randn(size))
    #x = randn(size)
    normalization = sum_constraint / sum(x)
    x .*= normalization
    return x
end

function solve_problem(A, y; L, ∇L!, Δt, λ=λ, sum_constraint=1/Δt, scale=1, n=(size(A, 2) - 1) ÷ scale_to_skip(scale) + 1, x_init=initialize_x(n; sum_constraint), loss_tol=0.01, max_itr=7000, ignore_warnings=false)

    @assert n == length(x_init)

    g = zeros(n) # gradient
    # @show sum_constraint
    # @show sum(x_init)
    # if !(sum(x_init) ≈ sum_constraint) #!isapprox(sum(x_init), sum_constraint; rtol=0.2)
    #     @warn "x_init does not sum to $sum_constraint, it sums to $(sum(x_init))...projecting x_init"
    #     proj!(x_init; sum_constraint)
    # end
    if scale != 1
        skip = scale_to_skip(scale)
        A = A[:, begin:skip:end] * skip
        #y = y / skip # NOT in place because we need a fresh copy of y in future calls
    end


    α=make_step_size(; A, y, Δt, λ, n, scale)

    x = x_init
    i = 1

    ∇L!(g, x; Δt, A, y, scale)

    loss_per_itr = Float64[]
    push!(loss_per_itr, L(x; Δt, A, y, λ, scale))

    #loss_tol *= sqrt(n) # scale by problem size

    while loss_per_itr[i] > loss_tol
        @. x -= α * g
        proj!(x; sum_constraint)
        #sum(x) ≈ sum_constraint || @error "Sum is not $sum_constraint, is $(sum(x))"
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

function solve_problem_multiscale(A, y; L, ∇L!, Δt, λ, sum_constraint=1/Δt,  n_scales=n_points_to_n_scales(size(A, 2)),x_init=initialize_x(3; sum_constraint= sum_constraint / scale_to_skip(n_scales)), loss_tol=0.01, max_itr=7000, ignore_warnings=false, show_plot=false)

    all_iterations = zeros(Int, n_scales)

    # α = make_step_size(;A,y,λ,Δt,n=size(A,2)) # Finest scale stepsize, use at every scale

    #y_copy = copy(y)

    # Coarsest scale solve
    #A_S, y_copy = scale_problem!(y_copy, A, y; grid=t, scale=n_scales)

    x_S, i_S, _ = solve_problem(A, y; L, ∇L!, λ, sum_constraint = sum_constraint / scale_to_skip(n_scales),
        x_init, ignore_warnings=true, max_itr=1, loss_tol=0, Δt, scale=n_scales) # force one gradient step
    # p = plot(coarsen(t, scale_to_skip(n_scales)), x_S)
    # Δt * scale_to_skip(n_scales)

    if show_plot; p = plot(range(-1,1,length=length(x_S)), x_S); end
    # @show sum(x_S)
    x_s = interpolate_solution(x_S)
    # @show sum(x_s)

    all_iterations[n_scales] = i_S

    # Middle scale solves
    for scale in (n_scales-1):-1:2 # Count down from larger to smaller scales
        #A_s, y_copy = scale_problem!(y_copy, A, y; grid=t, scale)
        x_s, i_s,_ = solve_problem(A, y; L, ∇L!, λ, sum_constraint = sum_constraint / scale_to_skip(scale),
            x_init=x_s, ignore_warnings=true, max_itr=1, loss_tol=0, Δt, scale) # force one gradient step
        # p = plot!(coarsen(t, scale_to_skip(scale)), x_s)
        # Δt = Δt * scale_to_skip(scale) # don't need if we use `scale`

        if show_plot; plot!(range(-1,1,length=length(x_s)), x_s); end
        x_s = interpolate_solution(x_s)

        all_iterations[scale] = i_s

        # @show opnorm(Symmetric(A_s*A_s'))
    end

    # Finest scale solve
    x_1, i_1, loss_per_itr = solve_problem(A, y; L, ∇L!, Δt, λ, sum_constraint, x_init=x_s, max_itr, loss_tol, scale=1, ignore_warnings)
    # p = plot!(t, x_1)
    # display(p)


    if show_plot; plot!(range(-1,1,length=length(x_1)), x_1); display(p); end

    all_iterations[1] = i_1

    return x_1, all_iterations, loss_per_itr
end

using Profile

profile = true

if profile
    n_scales = 4

    fine_scale_size = 2^n_scales + 1
    t = range(-1, 1, length=fine_scale_size)#[begin:end-1]
    Δt = Float64(t.step)

    A, x, y = make_problem(;t, σ)

    """Loss Function"""
    L(x; Δt=Δt, λ=λ, A=A, y=y,scale=1) = 0.5 * norm2(A*x .- y) + λ .* GL(x;Δt,scale)
    ∇L(x; Δt=Δt, λ=λ, A=A, y=y,scale=1) = A'*(A*x .- y) .+ λ .* ∇GL(x;Δt,scale)
    function ∇L!(z, x; Δt=Δt, λ=λ, A, y, scale=1)
        ∇GL!(z, x; Δt, scale)
        #z .*= λ
        #z .+= A' * (A * x .- y)
        # mul!(C, A, B, α, β) == ABα+Cβ
        mul!(z, A',  A*x .- y, 1, λ)
    end
    loss_tol = L(x) * (1 + percent_loss_tol) # 5e-4 # Good for scale=4, 9e-5
    #3.5e-5#2.9e-5#L(x; Δt) * (1 + percent_loss_tol) # 0.015

    # Compile
    xhat, n_itr_single, loss_per_itr_single = solve_problem(A, y; L, ∇L!, loss_tol, Δt, λ)
    xhat_multi, n_itr_multi, loss_per_multi = solve_problem_multiscale(A, y; L, ∇L!, loss_tol, Δt, λ, show_plot=true)

    x_naive = A \ y#A'y .* Δt
    # Plot a typical solution
    p = plot(; xlabel="t at scale $n_scales", ylabel="density")
    plot!(t, x; label="true distribution")
    plot!(t, x_naive; label="naive estimate")
    plot!(t, xhat;label="single scale")
    plot!(t, xhat_multi; label="multi-scale")
    display(p)

    # profile


    #@profview solve_problem(A, y; L, ∇L!, loss_tol, Δt, λ)
    #@profview solve_problem_multiscale(A, y; L, ∇L!, loss_tol, Δt, λ)
end

plot(loss_per_itr_single;yaxis=:log10)
plot(loss_per_multi)

show_summary = true
if show_summary
@show norm(∇L(xhat; Δt))
@show norm(∇L(xhat_multi; Δt))
@show L(xhat; Δt)
@show L(xhat_multi; Δt)
@show L(x; Δt)
@show n_itr_single
@show n_itr_multi
end
xhat, n_itr_single, loss_single = solve_problem(A, y; L, ∇L!, loss_tol, Δt, λ)
plot(t,x)
xhat_multi, n_itr_multi, loss_multi = solve_problem_multiscale(A, y; L, ∇L!, loss_tol, Δt, λ)
# @profview solve_problem(A, y; L, ∇L!, loss_tol, Δt, t, λ)
# @profview solve_problem_multiscale(A, y; L, ∇L!, loss_tol, Δt, t, λ)


for n_scales in 3:12
    fine_scale_size = 2^n_scales + 1
    t = range(-1, 1, length=fine_scale_size)#[begin:end-1]
    Δt = Float64(t.step)

    A, x, y = make_problem(;t, σ)

    """Loss Function"""
    L(x; Δt, λ=λ, A=A, y=y, scale=1) = 0.5 * norm2(A*x .- y) + λ .* GL(x;Δt,scale)
    println(L(x;Δt))
end

@time x_single, i_single, loss_single = solve_problem(A, y; L, ∇L!, loss_tol=7e-6, Δt, λ)
i_single
plot(t, x_single)
plot(loss_single; yaxis=:log10)
#7e-5

L(x;Δt)
L(x_single;Δt)
L(x_multi;Δt)
loss_single[end]
plot!(t,x)
@time x_multi, i_multi, loss_multi = solve_problem_multiscale(A, y; L, ∇L!, loss_tol=8e-5, Δt, λ)

plot(t, x_multi)
i_multi
plot(loss_multi;yaxis=:log10)

loss_tol = 4e-5
@benchmark solve_problem(A, y; L, ∇L!, loss_tol, Δt, λ)

@benchmark solve_problem_multiscale(A, y; L, ∇L!, loss_tol, Δt, λ)

∇L_scale(x, scale=1) = ∇L(x; Δt, A=A[:,begin:scale_to_skip(scale):end], y=y / scale_to_skip(scale), scale=1)
L_scale(x, scale=1) = L(x; Δt, A=A[:,begin:scale_to_skip(scale):end], y=y / scale_to_skip(scale), scale=1)

reconstruction(x; scale=1, skip=scale_to_skip(scale)) = 0.5*norm2(A[:,begin:skip:end] * x[begin:skip:end] * skip  - y)

AA(A;scale=1,skip=scale_to_skip(scale), A_=A[:,begin:skip:end]*skip) = opnorm(A_*A_') / skip
AA(A;scale=7)
1/Δt

∇GL_scale(x; scale=1,skip=scale_to_skip(scale)) =  laplacian_matrix(length(x[begin:skip:end]))*x[begin:skip:end]/Δt^t_power / skip

GL_scale(x; scale=1,skip=scale_to_skip(scale)) =  0.5*norm2(diff(x[begin:skip:end]))/Δt^t_power / skip

#######################
# Start of Benchmarks #
#######################

scales = 2:12 # 3:12
loss_tol_per_scale = zeros(length(scales))
#percent_loss_tol = 0.95 # old idea: iterate until loss * percent_los_tol < best_loss
# new idea: iterate until loss * percent_loss_tol < 99%-tile for 20 final losses


suite = BenchmarkGroup()

for (s, n_scales) in enumerate(scales)

    fine_scale_size = 2^n_scales + 1
    t = range(-1, 1, length=fine_scale_size)
    Δt = Float64(t.step)

    A, x, y = make_problem(; t, σ)

    """Loss Function"""
    L(x; Δt=Δt, λ=λ, A=A, y=y,scale=1) = 0.5 * norm2(A*x .- y) + λ .* GL(x;Δt,scale)
    ∇L(x; Δt=Δt, λ=λ, A=A, y=y,scale=1) = A'*(A*x .- y) .+ λ .* ∇GL(x;Δt,scale)
    function ∇L!(z, x; Δt=Δt, λ=λ, A, y, scale=1)
        ∇GL!(z, x; Δt, scale)
        #z .*= λ
        #z .+= A' * (A * x .- y)
        # mul!(C, A, B, α, β) == ABα+Cβ
        mul!(z, A', A*x .- y, 1, λ)
    end

    # loss_tol = 0 # run max iterations first # L(x; Δt) * (1 + percent_loss_tol) #0.015

    # # Compile
    # all_final_losses = Float64[]
    # xhat = similar(x) # give scope to these outside of the nested for loop
    # xhat_multi = similar(x)
    # for _ in 1:20
    #     xhat, _, loss_per_itr_single = solve_problem(A, y; L, ∇L!, loss_tol, Δt, λ)

    #     append!(all_final_losses, [loss_per_itr_single[end]])
    # end
    # loss_tol = maximum(all_final_losses)
    # loss_tol_per_scale[s] = loss_tol / percent_loss_tol

    # compile
    # loss_tol = 0.0005 * (1 + percent_loss_tol)
    loss_tol = L(x) * (1 + percent_loss_tol) # want xhat to be at least as good as our true x
    loss_tol_per_scale[s] = loss_tol
    xhat, itr_single, loss_per_itr_single = solve_problem(A, y; L, ∇L!, loss_tol, Δt, λ)
    xhat_multi, itr_multi, loss_per_itr_multi = solve_problem_multiscale(A, y; L, ∇L!, loss_tol, Δt, λ, show_plot=true)

    @show n_scales
    #@show 1 / opnorm(Symmetric(A'A) + λ/Δt^1*laplacian_matrix(length(x)))
    # @show opnorm(Symmetric(A'A))
    #@show ∇L(x)[end]
    #@show 1/Δt
    @show L(x)
    n=length(x)
    effective_gradient = make_step_size(;A,y,n,Δt,λ) .* ∇L(x)
    # top_val = top_eigenval(; A, y, Δt, λ, n, scale=1)
    # bot_val = bot_eigenval(; A, y, Δt, λ, n, scale=1)
    # @show top_val, bot_val
    # @show effective_gradient[end]
    # @show make_step_size(;A,y,n=length(x),Δt,λ)
    # @show ∇L(x)[end]
    @show itr_single
    # @show effective_gradient[end]
    @show itr_multi[1]

    #@show itr_single[1]
    # Plot a typical solution
    # p = plot(; xlabel="t at scale $n_scales", ylabel="density")
    # plot!(t, x; label="true distribution")
    # plot!(t, xhat;label="single scale")
    # plot!(t, xhat_multi; label="multi-scale")
    # display(p)

    # p = plot(;xlabel="# iterations at scale $n_scales", ylabel="loss", yaxis=:log10)
    # plot!(loss_per_itr_single[begin+2:end]; label="single scale")
    # plot!(loss_per_itr_multi[begin+2:end]; label="multi scale")
    # display(p)

    # Prep the benchmarks
    suite["single"][n_scales] = @benchmarkable solve_problem($A, $y; L=$L, ∇L! = $∇L!, loss_tol=$loss_tol, Δt=$Δt, λ=$λ)
    suite["multi"][n_scales] = @benchmarkable solve_problem_multiscale($A, $y; L=$L, ∇L! = $∇L!, loss_tol=$loss_tol, Δt=$Δt, λ=$λ)

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

# single_max_times = get_time(maximum, "single")
# single_min_times = get_time(minimum, "single")
# multi_max_times = get_time(maximum, "multi")
# multi_min_times = get_time(minimum, "multi")

top_quantile = 0.90
bot_quantile = 0.10

single_top_times = get_time(x -> quantile(x, top_quantile), "single")
single_bot_times = get_time(x -> quantile(x, bot_quantile), "single")
multi_top_times = get_time(x -> quantile(x, top_quantile), "multi")
multi_bot_times = get_time(x -> quantile(x, bot_quantile), "multi")

#single_median_times .*= 1e-6 # in ms
#multi_median_times .*= 1e-6 # in ms

problem_sizes = @. 2^scales + 1

p = plot(;
    xlabel="problem size (number of points)",
    ylabel="median time (ms)",
    xticks=(problem_sizes .- 1),
    xaxis=:log2,
    yaxis=:log10,
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

n_samples_single = [length(results["single"][S]) for S in scales]
n_samples_multi = [length(results["multi"][S]) for S in scales]

@show n_samples_single
@show n_samples_multi

end


# m, n = 6, 100
# A = randn(m, n)
# x = randn(n)
# y = randn(m)
# λ = 0.001

# z = randn(n)

# @benchmark f6(z) setup=(z=randn(n))

# function f1(z)
#     return A' * (A * x .- y) .+ λ .* z
# end

# function f2(z)
#     z .*= λ # TODO speed this up with mul!(C, A, B, α, β) == ABα+Cβ
#     z .+= A' * (A * x .- y)
# end


# function f3(z) #winner so far
#     z .= A' * (A * x .- y) .+ λ .* z
# end

# function f4(z)
#     z .= A' * (A * x - y) + λ * z
# end

# function f5(z)
#     z .= A' * A * x - A'*y + λ * z
# end


# function f6(z)
#     mul!(z, A',  A*x .- y, 1, λ)
# end

# f6(z)
# f6(z)
# f3(z)
