# generate synthetic data for KDE and tensor factorization

using Random
using KernelDensity
using Pkg
Pkg.add("Distributions")
Pkg.add("Plots")

using Distributions
using MatrixTensorFactor
using Plots


I = 5 # number mixtures
J = 32 # Number of samples in the x dimention
K = 32 # Number of samples in the y dimention

#Random.seed!(123)

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
x = range(-10,10,length=J)
Δx = x[2] - x[1]

# y values to sample the densities at
y = range(-10,10,length=K)
Δy = y[2] - y[1]

# Collect sample points into a matrix
xy = Matrix{Vector{Float64}}(undef, length(y),length(x))
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
I = length(distributions)

# Collect into a tensor that is size(Y) == (Sinks x Features x Samples)
sinks = [pdf.((d,), xy) for d in distributions]
Y = hcat(sinks...)
Y = reshape(Y, (length(x),length(y),length(distributions)))
Y = permutedims(Y, (3,1,2))
Y .*= Δx * Δy # Scale factors
sum.(eachslice(Y, dims=1)) # should all be 1

# get singular values to find optimal rank
Y_mat = reshape(Y, I, :)
σ = svdvals(Y_mat)
plot(σ)
partial_sum = [sum(σ[i:I].^2) .^ 0.5 for i in 2:I]
push!(partial_sum, 0)
plot(partial_sum, label="I-r smallest singular values norm", xlabel="rank", linewidth=2)

# Perform decomposition
final_loss = zeros(I)
for r in 1:I
C, F, rel_errors, norm_grad, dist_Ncone = nnmtf(Y, r;
    tol=1e-5 / sqrt(R*(I+J*K)),
    projection=:nnscale,
    normalize=:slices,
    stepsize=:lipshitz,
    momentum=true,
    delta=0.8,
    criterion=:ncone,
    online_rank_estimation=true)

    final_loss[r] = norm(Y - C*F)
end

plot!(final_loss, label="Frobenius norm of relative error tensor")

R = 3 # Can see that R=3 is optimal
C, F, rel_errors, norm_grad, dist_Ncone = nnmtf(Y, R;
    tol=1e-5 / sqrt(R*(I+J*K)),
    projection=:nnscale,
    normalize=:slices,
    stepsize=:lipshitz,
    momentum=true,
    delta=0.8,
    criterion=:ncone,
    online_rank_estimation=true)

@show (I, R, J, K)
@show length(rel_errors)
@show mean_rel_error(C*F, Y; dims=1)

heatmap(x, y, Y[1,:,:], title="True Y slice 1") |> display
heatmap(x, y, (C*F)[1,:,:], title="learned Y slice 1") |> display

F ./= Δx * Δy # Rescale factors

# Plot learned factors
heatmap(C, yflip=true, title="Learned Coefficients", clims=(0,1)) |> display
heatmap(C_true, yflip=true, title="True Coefficients", clims=(0,1)) |> display # possibly permuted order

for r in 1:R
    heatmap(x, y, F[r,:,:], title="Learned Source $r") |> display
end

# Plot convergence
plot(rel_errors[2:end], yaxis=:log10) |> display
plot(norm_grad[2:end], yaxis=:log10) |> display
plot(dist_Ncone[2:end], yaxis=:log10) |> display
