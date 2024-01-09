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
source2a = Normal(-2, 3)
source3a = Normal(0, 1)
source1b = Uniform(-7, 2)
source2b = Exponential(2)
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

# One mixture density
heatmap(x,y, pdf.((distribution5,), xy)) |> display

distributions = [distribution1, distribution2, distribution3, distribution4, distribution5]

# Collect into a tensor that is size(Y) == (Sinks x Features x Samples)
sinks = [pdf.((d,), xy) for d in distributions]
Y = hcat(sinks...)
Y = reshape(Y, (length(x),length(y),length(distributions)))
Y = permutedims(Y, (3,1,2))
Y .*= Δx * Δy # Scale factors
sum.(eachslice(Y, dims=1)) # should all be 1

# Perform decomposition
C, F, rel_errors, norm_grad, dist_Ncone = nnmtf(Y, R, tol=1e-6, projection=:nnscale, normalize=:slices, stepsize=:spg, criterion=:ncone)
F ./= Δx * Δy # Rescale factors

# Plot learned factors
heatmap(C, yflip=true, title="Learned Coefficients") |> display
heatmap(C_true, yflip=true, title="True Coefficients") |> display # possibly permuted order

heatmap(x, y, F[1,:,:], title="Learned Source 1") |> display
heatmap(x, y, F[2,:,:], title="Learned Source 2") |> display
heatmap(x, y, F[3,:,:], title="Learned Source 3") |> display

# Plot convergence
plot(rel_errors[2:end], yaxis=:log10) |> display
plot(norm_grad[2:end], yaxis=:log10) |> display
plot(dist_Ncone[2:end], yaxis=:log10) |> display
