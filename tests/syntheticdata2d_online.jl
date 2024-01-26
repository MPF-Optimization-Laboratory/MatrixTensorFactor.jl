# generate synthetic data for KDE and tensor factorization

using Random
using KernelDensity
using Distributions
using MatrixTensorFactor
using Plots

Random.seed!(123)

#############################################
#   NNMTF - 2 Features - Perfect Densities  #
#############################################

# Three sources, product distributions
R = 3
source1a = Normal(5, 1)
source1b = Uniform(-7, 2)
source2a = Normal(-2, 3)
source2b = Exponential(2)
source3a = Normal(0, 1)
source3b = Normal(0, 3)

source1 = product_distribution([source1a, source1b])
source2 = product_distribution([source2a, source2b])
source3 = product_distribution([source3a, source3b])

sources = (source1, source2, source3)

# x values to sample the densities at
x = range(-10,10,length=30)
Δx = x[2] - x[1]

# y values to sample the densities at
y = range(-10,10,length=30)
Δy = y[2] - y[1]

# Collect sample points into a matrix
xy = Matrix{Vector{Float64}}(undef, length(x),length(y))
for (i, x) in enumerate(x)
    for (j, y) in enumerate(y)
        xy[j, i] = [x, y]
    end
end

# Plot the true sources
heatmap(x,y,pdf.((source1,), xy)) |> display
heatmap(x,y,pdf.((source2,), xy)) |> display
heatmap(x,y,pdf.((source3,), xy)) |> display

# Generate mixing matrix
p1 = [0, 0.4, 0.6]
p2 = [0.3, 0.3, 0.4]
p3 = [0.8, 0.2, 0]
p4 = [0.2, 0.7, 0.1]
p5 = [0.6, 0.1, 0.3]

C_true = hcat(p1,p2,p3,p4,p5)'

# Generate mixing distributions
distribution1 = MixtureModel([sources...], p1)
distribution2 = MixtureModel([sources...], p2)
distribution3 = MixtureModel([sources...], p3)
distribution4 = MixtureModel([sources...], p4)
distribution5 = MixtureModel([sources...], p5)

make_samples(distribution, N=1000) = [rand(distribution) for _ in 1:N]

# Example heatmap from distribution5
samples5 = make_samples(distribution5)
f = kde(hcat(samples5...)')
heatmap(f.x, f.y, f.density')

# True pdf of same mixture
heatmap(x,y, pdf.((distribution5,), xy)) |> display

# Make an initial Y with limited number of samples
N = 10 # number of distribution samples in batch
K = 128 # number of KDE samples in both x & y dimentions
distributions = [distribution1, distribution2, distribution3, distribution4, distribution5]

xs = range(-10, 10, length=K)
ys = range(-10, 10, length=K)
Δx = xs[2]-xs[1]
Δy = ys[2]-ys[1]
xy = Matrix{Vector{Float64}}(undef, length(x),length(y))
for (i, x) in enumerate(x)
    for (j, y) in enumerate(y)
        xy[j, i] = [x, y]
    end
end

samples = make_samples.(distributions, N)
fs = [kde(hcat(x...)', (xs, ys); bandwidth=(0.5,0.5)).density' for x in samples] # Need to sample at the same place with the sample bandwith

heatmap(fs[1])

# Collect into a tensor that is size(Y) == (Sinks x Features x Samples)
Y = cat(fs...;dims=3)
Y = permutedims(Y, (3,1,2))
Y .*= Δx * Δy
sum.(eachslice(Y, dims=1))
heatmap(xs, ys, Y[1,:,:])

# May have to extent KernelDensity so I can add one sample to an estimate (with appropriate bandwidth)
#=
function updateY!(Y, new_sample; N, I, xs, ys)
    # Turn new_sample point into a gaussian over xs, ys
    a, b = new_sample
    σ = N^(-0.2) #bandwidth should get small as N gets big
    kernel = @. (2π*σ^2)^(-1)*exp(-0.5*((xs'-a)^2 + (ys-b)^2)/σ^2)
    kernel ./= sum(kernel)

    Y[I, :, :] .= (Y[I, :, :] .* N .+ kernel) ./ (N+1)
end
=#
#updateY!(Y, (-5,0); N=13, I=1, xs, ys)
#heatmap(xs, ys, Y[1,:,:])

# Make function that takes one more sample from a distribution,
# and updates Y, the current best guess for the KDE

# Perform decomposition
C, F, rel_errors, norm_grad, dist_Ncone = nnmtf_proxgrad_online(Y, 3, distributions, N;
    xs, ys,
    tol=1e-8,
    projection=:nnscale,
    normalize=:slices,
    stepsize=:lipshitz,
    momentum=false,
    criterion=:ncone)
F ./= Δx * Δy # Rescale factors

# Plot learned factors
heatmap(C, yflip=true, title="Learned Coefficients") |> display
heatmap(C_true, yflip=true, title="True Coefficients") |> display # possibly permuted order

heatmap(xs, ys, F[1,:,:], title="Learned Source 1") |> display
heatmap(xs, ys, F[2,:,:], title="Learned Source 2") |> display
heatmap(xs, ys, F[3,:,:], title="Learned Source 3") |> display

# Plot convergence
plot(rel_errors[2:end], yaxis=:log10) |> display
plot(norm_grad[2:end], yaxis=:log10) |> display
plot(dist_Ncone[2:end], yaxis=:log10) |> display
