# Test of multiscale

using BlockTensorFactorization
using HDF5
using BenchmarkTools
using Logging
using Random

Random.seed!(314)

filename = "./test/geomultiscaletest/geodata.h5"

myfile = h5open(filename, "r")

#Y = myfile["Y_257_points"] |> read
Y = myfile["Y_1025_points"] |> read

close(myfile)

size(Y)

Y_backup = copy(Y)

# # Add an extra slice so the continuous dimension is one plus a power of 2
# Y = cat(Y, zeros(20, 7, 1) ;dims=3)

scaleB_rescaleA! = ConstraintUpdate(0, l1scale_average12slices! ∘ nonnegative!;
    whats_rescaled=(x -> eachcol(factor(x, 1)))
)

scaleA_rows! = ConstraintUpdate(1, l1scale_rows! ∘ nonnegative!;
    whats_rescaled=nothing
)

nonnegativeA! = ConstraintUpdate(1, nonnegative!)

nonnegativeB! = ConstraintUpdate(0, nonnegative!)
# [l1scale_average12slices! ∘ nonnegative!, nonnegative!]
# [scaleB_rescaleA!, scaleA_rows!]
# [nonnegativeB!, scaleA_rows!]

options = (
    rank=3,
    momentum=false,
    model=Tucker1,
    tolerance=(0.12),
    converged=(RelativeError),
    do_subblock_updates=false,
    constrain_init=true,
    constraints=[l1scale_average12slices! ∘ nonnegative!, nonnegative!],
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError],
    maxiter=200
)

#[l1scale_average12slices! ∘ nonnegative!,nonnegative!]

println("First Compile Time")
@time decomposition, stats_data, kwargs = factorize(Y; options...);

@time decomposition, stats_data, kwargs = multiscale_factorize(Y;
    continuous_dims=3,options...);
println("")
println("Regular Run")
@time decomposition, stats_data, kwargs = factorize(Y; options...);

@time decomposition, stats_data, kwargs = multiscale_factorize(Y;
    continuous_dims=3,options...);

println("Benchmark Run")


logger = ConsoleLogger(stdout, Logging.Warn)
global_logger(logger)

bmk = @benchmark factorize(Y; options...)
display(bmk)
bmk = @benchmark multiscale_factorize(Y; continuous_dims=3,options...)
display(bmk)

logger = ConsoleLogger(stdout, Logging.Info)
global_logger(logger);
