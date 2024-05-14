"""
Goal of this test is to compare different projections and stepsizes on different shaped
data. Specificaly, if the nnscale method works better than projection when the matrix Y
being factored is very tall, very wide, or square.
"""

using Random
using MatrixTensorFactor
using Pkg; Pkg.add("Plots"); using Plots
using Statistics

sizes = [
    (80, 5),
    (40, 10),
    (20, 20),
    (10, 40),
    (5, 80),
]

I, J = 40, 10

function weird_check(v)
    @show all(isreal.(v))
    @show all(v .> 0)
    @show all(isfinite.(v))
end

# Make the test
function nnmtf_test(projection; projectionA=projection, projectionB=projection, Y, R)
    C, F, rel_errors, norm_grad, dist_Ncone = nnmtf(Y, R;
        maxiter=300,
        tol=1e-32,
        momentum=false,
        projection,
        normalize=:fibres,
        rescale_Y=false,
        stepsize=:lipshitz, #:lipshitz
        online_rank_estimation=false,
        projectionA,
        projectionB,
        delta=0.9) # momentum parameter between 0 and 1

    # n_iterations = length(rel_errors)
    # final_rel_error = rel_errors[end]
    # @show projection, n_iterations, final_rel_error

    F = dropdims(F; dims=2)

    # # Plot learned factors
    # heatmap(C, yflip=true, title="Learned C with ($projectionA, $projectionB)") |> display
    # heatmap(A, yflip=true, title="True C") |> display # possibly permuted order

    # p=heatmap(F[:,1,:], title="Learned Sources")
    # for r in 2:R
    #     plot!(x, F[r,:])
    # end
    # display(p)

    # # Plot convergence
    # plot(rel_errors[2:end], yaxis=:log10, title="Relative Error with ($projectionA, $projectionB)") |> display
    # plot(norm_grad[2:end], yaxis=:log10, title="Norm of Gradient with ($projectionA, $projectionB)") |> display
    # plot(dist_Ncone[2:end], yaxis=:log10, title="Distance between -Gradient\n and Normal Cone with ($projectionA, $projectionB)") |> display

    return rel_errors, C, F
end

function main(I,J)
    # Make factors
    # K = min(I,J)
    R = 3

    init(x...) = abs.(rand(x...))

    A = init(I, R)
    A = A ./ sum.(eachrow(A))
    B = init(R, J)
    B = B ./ sum.(eachrow(B))

    # Construct the tensor Y
    Y_mat = A*B

    Y = reshape(Y_mat, (size(Y_mat)...,1))
    Y = permutedims(Y, (1,3,2))
    @assert all(sum.(eachslice(Y, dims=(1,2))) .≈ 1)

    #heatmap(A)
    @assert all(sum.(eachrow(A)) .≈ 1)

    #heatmap(B)
    @assert all(sum.(eachrow(B)) .≈ 1)

    # Perform the test
    objective_simplex, _, _ = nnmtf_test(:simplex; Y, R) # standard proxgrad
    objective_nnscale, _, _ = nnmtf_test(:nnscale; Y, R) # our rescaling trick
    objective_simplexB, _, _ = nnmtf_test(:simplex; projectionA=:nonnegative, Y, R) # proxgrad only enforce B is in simplex
    objective_simplexA, _, _ = nnmtf_test(:simplex; projectionB=:nonnegative, Y, R) # proxgrad only enforce A is in simplex

    # p = plot(objective_nnscale[2:end], yaxis=:log10, label="nnscale",
    #     xlabel="iteration", ylabel="relative error")
    # plot!(objective_simplex[2:end], yaxis=:log10, label="simplex A & B")
    # plot!(objective_simplexA[2:end], yaxis=:log10, label="simplex A only")
    # plot!(objective_simplexB[2:end], yaxis=:log10, label="simplex B only")
    # title!("size of Y is (I, J)=$((I, J))")
    # display(p)

    return objective_nnscale, objective_simplex, objective_simplexA, objective_simplexB
end

N_trials = 50
N_methods = 4 #nnscale, simplex, simplexA, simplexB
trial_names = ["nnscale", "simplex", "simplexA", "simplexB"]
trial_colours = [:blue, :green, :orange, :purple]

Random.seed!(100)
for (I,J) in sizes
    # Initialize the 4 arrays that will each hold N_trials of the objective's decent
    outs = [Vector{Float64}[] for _ in 1:N_methods]

    for _ in 1:N_trials
        out = main(I, J) # out = nnscale, simplex, simplexA, simplexB
        out = (f[2:end] for f in out) # remove initilization
        [push!(Xs, X) for (Xs, X) in zip(outs, out)]
    end

    nnscales, simplexs, simplexAs, simplexBs = outs

    mkstat(f) = [f.(zip(Xs...)) for Xs in outs]

    means = mkstat(mean)
    medians = mkstat(median)
    maxs = mkstat(x -> quantile(x, 0.75))
    mins = mkstat(x -> quantile(x, 0.25))
    stds = mkstat(std)

    p = plot()

    for (i, (name, colour)) in enumerate(zip(trial_names, trial_colours))
        #i == 1 ? continue : nothing
        plot!(medians[i];
            linewidth=2,
            yaxis=:log10,
            ribbon=(medians[i]-mins[i], maxs[i]-medians[i]),#stds[i],#
            fillalpha=0.2,
            label=name,
            colour
        )
        #plot!(mins[i]; linestyle=:dash, label=name*" maxs", colour)
    end
    title!("size(Y)=$((I, J)), $N_trials trials")
    display(p)
end
