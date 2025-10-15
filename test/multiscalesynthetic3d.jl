# generate synthetic data for KDE and tensor factorization

using Random
using KernelDensity
#using Pkg
#Pkg.add("Distributions")
#Pkg.add("Plots")
#Pkg.add("PlotlyJS")

using Distributions
using BlockTensorFactorization
using Plots
using PlotlyJS

show_plots = false

J = 65 # Number of samples in the x dimension
K = 65 # Number of samples in the y dimension
L = 65 # Number of samples in the z dimension

Random.seed!(123)

# Three sources, product distributions
R = 3
source1a = Normal(4, 1)
source1b = Uniform(-7, 2)
source1c = Uniform(-1, 1)

source2a = Normal(0, 3)
source2b = Uniform(-2, 2)
source2c = Exponential(2)

source3a = Exponential(1)
source3b = Normal(0, 1)
source3c = Normal(0, 3)

source1 = product_distribution([source1a, source1b, source1c])
source2 = product_distribution([source2a, source2b, source2c])
source3 = product_distribution([source3a, source3b, source3c])

sources = (source1, source2, source3)

# x values to sample the densities at
x = range(-10,10,length=J)
Δx = x[2] - x[1]

# y values to sample the densities at
y = range(-10,10,length=K)
Δy = y[2] - y[1]

# z values to sample the densities at
z = range(-10,10,length=K)
Δz = z[2] - z[1]

# Collect sample points into a matrix
xyz = Array{Vector{Float64}, 3}(undef, length(y),length(x), length(z))
for (j, x) in enumerate(x)
    for (k, y) in enumerate(y)
        for (l, z) in enumerate(z)
            xyz[j, k, l] = [x, y, z]
        end
    end
end

# Plot the true sources
XXX, YYY, ZZZ = mgrid(x, y, z)
source1_density = pdf.((source1,), xyz)
source2_density = pdf.((source2,), xyz)
source3_density = pdf.((source3,), xyz);

function plot3d(pdf3d; isomin=1e-4)
    PlotlyJS.plot(isosurface(
        x=XXX[:], # isosurface wants all entries in a single list
        y=YYY[:],
        z=ZZZ[:],
        value=pdf3d[:],
        opacity=0.6,
        isomin=isomin,
        isomax=maximum(pdf3d)*0.9,
        surface_count=4, # number of isosurfaces, 2 by default: only min and max
        colorbar_nticks=4, # colorbar ticks correspond to isosurface values
        caps=attr(x_show=false, y_show=false),
    )) |> display
end
if show_plots
    plot3d(source1_density)
    plot3d(source2_density)
    plot3d(source3_density)
end

#heatmap(x,y,pdf.((source1,), xyz)) |> display


#heatmap(x,y,pdf.((source2,), xy)) |> display
#heatmap(x,y,pdf.((source3,), xy)) |> display

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
#heatmap(x,y, pdf.((distribution5,), xy)) |> display
#plot3d(pdf.((distribution3,), xyz))


distributions = [distribution1, distribution2, distribution3, distribution4, distribution5]
I = length(distributions)

# Collect into a tensor that is size(Y) == (Sinks x Features x Samples)
sinks = [pdf.((d,), xyz) for d in distributions]
Y = cat(sinks...; dims=4)
#Y = reshape(Y, (length(x),length(y),length(distributions)))
#Plots.histogram(Y[:]; xaxis=:log10, yaxis=:log10)
Y = permutedims(Y, (4,1,2,3))
#plot3d(Y[1,:,:,:];isomin=1e-3)
Y .*= Δx * Δy * Δz # Scale factors
Y_slices = eachslice(Y, dims=1)
correction = sum.(Y_slices) # should all be 1, might be 0.9783... from truncating and discretizing pdfs
Y_slices ./= correction

# Perform decomposition
# C, F, rel_errors, norm_grad, dist_Ncone = nnmtf(Y, R;
#     tol=1e-5 / sqrt(R*(I+J*K*L)),
#     projection=:nnscale,
#     normalize=:slices,
#     stepsize=:lipshitz,
#     momentum=true,
#     #momentum=false,
#     delta=0.8,
#     criterion=:ncone,
#     online_rank_estimation=true)
scaleB_rescaleA! = ConstraintUpdate(0, l1scale_1slices! ∘ nonnegative!;
    whats_rescaled=(x -> eachcol(factor(x, 1)))
)
nonnegativeB! = ConstraintUpdate(0, nonnegative!)
nonnegativeA! = ConstraintUpdate(1, nonnegative!)
#[l1scale_1slices! ∘ nonnegative!, nonnegative!]

options = (
    rank=3,
    momentum=true,
    model=Tucker1,
    tolerance=(1e-5),
    converged=(GradientNNCone),
    do_subblock_updates=false,
    constrain_init=true,
    constraints=[scaleB_rescaleA!, nonnegativeA!],
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError],
    maxiter=200
)

# First pass to compile
@time decomposition, stats_data, kwargs = multiscale_factorize(Y; continuous_dims=[2, 3, 4], options...);

@time decomposition, stats_data, kwargs = factorize(Y; options...);

# Second pass to time

using BenchmarkTools
using BenchmarkPlots
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 20 # default 5
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = false # default false
# t_multi = @btime decomposition, stats_data, kwargs = multiscale_factorize(Y; continuous_dims=[2, 3, 4], options...);

# t_single = @btime decomposition, stats_data, kwargs = factorize(Y; options...);


# bmk = @benchmark factorize(Y; options...)
# display(bmk)


bmk = @benchmarkable multiscale_factorize(Y; continuous_dims=[2, 3, 4], options...) samples=5
the_result = run(bmk)
StatsPlots.plot(the_result)
display(bmk)

# @show (I, R, J, K, L)
# @show length(rel_errors)
# @show mean_rel_error(C*F, Y; dims=1)

# # BlockTensorFactorization.Core.expand_decomposition_constraints(Y, options)[2]
# # BlockedUpdate([BlockTensorFactorization.Core.scale_decomposition_constraint([2,3,4], constraint, 2, Tucker1(size(Y),3)) for constraint in BlockTensorFactorization.Core.expand_decomposition_constraints(Y, options)[2]])
# constraints = nothing
# kwargs=options
# kwargs_copy = nothing
# expanded_constraints = nothing
# begin
#     continuous_dims, kwargs = BlockTensorFactorization.Core.initialize_continuous_dims(Y; kwargs...)
#     scales, kwargs = BlockTensorFactorization.Core.initialize_scales(Y; kwargs...)
#     coarsest_scale, finer_scales... = scales

#     # Factorize Y at the coarsest scale
#     Yₛ = BlockTensorFactorization.Core.coarsen(Y, coarsest_scale; dims=continuous_dims, kwargs...)

#     #constraints, kwargs = BlockTensorFactorization.Core.scale_constraints(Yₛ, coarsest_scale; continuous_dims, kwargs...)

#     decomposition, constraints = BlockTensorFactorization.Core.expand_decomposition_constraints(Yₛ, kwargs)

#     any(x-> typeof(decomposition) <: x,  BlockTensorFactorization.Core.IMPLEMENTED_DECOMPOSITION_CONSTRAINT_SCALING)

#     constraints = BlockedUpdate([BlockTensorFactorization.Core.scale_decomposition_constraint(continuous_dims, constraint, S, decomposition) for constraint in constraints])

#     #  kwargs = Dict{Symbol,Any}(kwargs)

#     #  kwargs[:continuous_dims] = continuous_dims
#     #  S = log2(coarsest_scale) # TODO Don't assume the scale is a power of 2


#     #  kwargs_copy = deepcopy(kwargs) # Don't mess up anything since the following functions mutate kwargs
#     #  kwargs_copy = BlockTensorFactorization.Core.default_kwargs(Yₛ; kwargs_copy...) # TODO Is there some way to clean this up?
#     #  decomposition, kwargs_copy = BlockTensorFactorization.Core.initialize_decomposition(Yₛ; kwargs_copy...)


#     #  expanded_constraints = BlockTensorFactorization.Core.parse_constraints(kwargs_copy[:constraints], decomposition; kwargs_copy...)


# end


# #heatmap(x, y, Y[1,:,:], title="True Y slice 1") |> display
# #heatmap(x, y, (C*F)[1,:,:], title="learned Y slice 1") |> display

# F ./= Δx * Δy * Δz # Rescale factors

# Plot learned factors

# Plots.heatmap(C, yflip=true, title="Learned Coefficients", clims=(0,1)) |> display
# Plots.heatmap(C_true, yflip=true, title="True Coefficients", clims=(0,1)) |> display # possibly permuted order

# F = factor(decomposition, 0)


# for r in 1:R
#     if r != 3
#         continue
#     end
#     plot3d(F[r, :, :, :]) |> display
#     #heatmap(x, y, F[r,:,:], title="Learned Source $r") |> display
# end

# plot3d(F[2, :, :, :])

# #plot3d(Y[3, :, :, :])

# # Plot convergence
# Plots.plot(rel_errors[2:end], yaxis=:log10) |> display
# Plots.plot(norm_grad[2:end], yaxis=:log10) |> display
# Plots.plot(dist_Ncone[2:end], yaxis=:log10) |> display


# ########## tucker-2 decomposition
# #=
# B_permuted = permutedims(F, (3,1,2))

# CB, FB, rel_errors, norm_grad, dist_Ncone = nnmtf(B_permuted, 32;
#     tol=1e-5 / sqrt(R*(I+J*K)),
#     projection=:nnscale,
#     normalize=:slices,
#     stepsize=:lipshitz,
#     momentum=true,
#     delta=0.8,
#     criterion=:ncone,
#     online_rank_estimation=true)
# =#
