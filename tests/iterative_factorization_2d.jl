# generate synthetic data for KDE and tensor factorization

using Random
using KernelDensity
using Pkg
Pkg.add("Distributions")
Pkg.add("Plots")

using Distributions
using MatrixTensorFactor
using Plots

Random.seed!(1234)

function generate(J,K)

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
    p1 = [0.0, 0.4, 0.6]
    p2 = [0.3, 0.3, 0.4]
    p3 = [0.8, 0.2, 0.0]
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
    return I,R,x,y,C_true,Y,Δx,Δy
end

################################################################################
############### Run using a small 16 x 16 grid initilization ###################
################################################################################


J = 16 # Number of samples in the x dimention
K = 16 # Number of samples in the y dimention

I,R,x,y,C_true,Y,Δx,Δy = generate(J,K)

# Perform decomposition
options = (
    :tol => 0.02,  # 98%   #1e-6 #/ sqrt(R*(I+J*K)),
    :projection => :nnscale,
    :normalize => :slices,
    :stepsize => :lipshitz,
    #:stepsize => :spg,
    :momentum => true,
    #:momentum => false,
    :delta => 0.8,
    :criterion => :relativeerror, #:ncone,
)
@time C, F, rel_errors, norm_grad, dist_Ncone = nnmtf(Y, R; options...);

#@show (I, R, J, K)
@show length(rel_errors);
@show mean_rel_error(C*F, Y; dims=1);

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

################################################################################
########## Re Run using an upscaled "F" and "C" as the initilization ###########
################################################################################

using Images

F .*= Δx * Δy # Rescale factors
J = 512 # Number of samples in the x dimention
K = 512 # Number of samples in the y dimention
F_upscaled = imresize(F,(3, J, K))


I,R,x,y,C_true,Y,Δx,Δy = generate(J,K)

for r in 1:R
    heatmap(x, y, F_upscaled[r,:,:], title="Upscaled Learned Source $r") |> display
end

@time C, F, rel_errors, norm_grad, dist_Ncone = nnmtf(Y, R; A_init=C, B_init=F_upscaled, options...)

@show length(rel_errors);
@show mean_rel_error(C*F, Y; dims=1);
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

##################################################################
########### Fresh Solve on large grid #########################
###############################################################

@time C, F, rel_errors, norm_grad, dist_Ncone = nnmtf(Y, R; options...)


@show length(rel_errors);
@show mean_rel_error(C*F, Y; dims=1);
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
