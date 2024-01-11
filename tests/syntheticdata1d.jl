# generate synthetic data for KDE and tensor factorization

using Random
using KernelDensity
using Distributions
using MatrixTensorFactor
using Plots

Random.seed!(123)

####################
#   Quick KDE 1D   #
####################

N = 10 # number of samples
M = 3 # features

list_of_samples = [randn(M) for _ in 1:N]

list_of_measurements = [[n for n in m] for m in zip(list_of_samples...)]

kde(list_of_measurements[1])

kdes = make_densities(list_of_samples)

f = kdes[1] # the first estimated kernel

p = plot(f.x, f.density)
display(p)

############################################
#   NNMTF - 1 Feature - Perfect Densities  #
############################################

# Three sources, all normal
R = 3
source1 = Normal(5, 1)
source2 = Normal(-4, 2)
source3 = Normal(0, 1)

sources = (source1, source2, source3)

# x values to sample the densities at
x = range(-10,10,length=100)
Δx = x[2] - x[1]

# Plot the true sources
p = plot(x, pdf.((source1,), x))
plot!(x, pdf.((source2,), x))
plot!(x, pdf.((source3,), x))
display(p)

# Generate mixing matrix
p1 = [0, 0.4, 0.6]
p2 = [0.3, 0.3, 0.4]
p3 = [0.9, 0.1, 0]
p4 = [0.2, 0.7, 0.1]
p5 = [0.6, 0.1, 0.3]

C_true = hcat(p1,p2,p3,p4,p5)'

# Generate mixing distributions
distribution1 = MixtureModel(Normal[sources...], p1)
distribution2 = MixtureModel(Normal[sources...], p2)
distribution3 = MixtureModel(Normal[sources...], p3)
distribution4 = MixtureModel(Normal[sources...], p4)
distribution5 = MixtureModel(Normal[sources...], p5)

# One mixture density
plot(x, pdf.((distribution2,), x)) |> display

distributions = [distribution1, distribution2, distribution3, distribution4, distribution5]

# Collect into a tensor that is size(Y) == (Sinks x Features x Samples)
sinks = [pdf.((d,), x) for d in distributions]
Y = hcat(sinks...)
Y = reshape(Y, (size(Y)...,1))
Y = permutedims(Y, (2,3,1))
Y .*= Δx # Scale factors
sum.(eachslice(Y, dims=(1,2))) # should all be 1

function nnmtf_test(projection)
    C, F, rel_errors, norm_grad, dist_Ncone, R = nnmtf(Y;
        maxiter=500, tol=1e-10, momentum=true,projection, normalize=:fibres, rescale_Y=false, stepsize=:lipshitz)

    n_iterations = length(rel_errors)
    final_rel_error = rel_errors[end]
    @show projection, n_iterations, final_rel_error

    F = dropdims(F; dims=2)
    F ./= Δx # Rescale factors

    # Plot learned factors
    heatmap(C, yflip=true, title="Learned Coefficients with $projection") |> display
    heatmap(C_true, yflip=true, title="True Coefficients with $projection") |> display # possibly permuted order

    p=plot(x, F[1,:], title="Learned Sources")
    for r in 2:R
        plot!(x, F[r,:])
    end
    display(p)

    # Plot convergence
    plot(rel_errors[2:end], yaxis=:log10, title="Relative Error with $projection") |> display
    plot(norm_grad[2:end], yaxis=:log10, title="Norm of Gradient with $projection") |> display
    plot(dist_Ncone[2:end], yaxis=:log10, title="Distance between -Gradient\n and Normal Cone with $projection") |> display

    return rel_errors
end

objective_simplex = nnmtf_test(:simplex) # standard proxgrad
objective_nnscale = nnmtf_test(:nnscale) # our rescaling trick

plot(objective_simplex[2:end], yaxis=:log10, label="simplex",
    xlabel="iteration", ylabel="relative error")
plot!(objective_nnscale[2:end], yaxis=:log10, label="nnscale")
