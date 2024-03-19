"""
Goal of this test is to compare different projections and stepsizes on different shaped
data. Specificaly, if the nnscale method works better than projection when the matrix Y
being factored is very tall, very wide, or square.
"""

using Random
using MatrixTensorFactor
using Plots

sizes = [
    (80, 5),
    (40, 10),
    (20, 20),
    (10, 40),
    (5, 80),

]
#I, J = 40, 10

for (I,J) in sizes
# Make factors
Random.seed!(100)

K = min(I,J)
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

# Make the test
function nnmtf_test(projection; projectionA=projection, projectionB=projection)
    C, F, rel_errors, norm_grad, dist_Ncone = nnmtf(Y, R;
        maxiter=150,
        tol=1e-15,
        momentum=false,
        projection,
        normalize=:fibres,
        rescale_Y=false,
        stepsize=:lipshitz, #:lipshitz
        online_rank_estimation=false,
        projectionA,
        projectionB,
        delta=0.9) # momentum parameter between 0 and 1

    n_iterations = length(rel_errors)
    final_rel_error = rel_errors[end]
    @show projection, n_iterations, final_rel_error

    F = dropdims(F; dims=2)

    # Plot learned factors
    #heatmap(C, yflip=true, title="Learned C with ($projectionA, $projectionB)") |> display
    #heatmap(A, yflip=true, title="True C") |> display # possibly permuted order

    #p=heatmap(F[:,1,:], title="Learned Sources")
    #for r in 2:R
    #    plot!(x, F[r,:])
    #end
    #display(p)

    # Plot convergence
    #plot(rel_errors[2:end], yaxis=:log10, title="Relative Error with ($projectionA, $projectionB)") |> display
    #plot(norm_grad[2:end], yaxis=:log10, title="Norm of Gradient with ($projectionA, $projectionB)") |> display
    #plot(dist_Ncone[2:end], yaxis=:log10, title="Distance between -Gradient\n and Normal Cone with ($projectionA, $projectionB)") |> display

    return rel_errors, C, F
end

# Perform the test
objective_simplex, _, _ = nnmtf_test(:simplex) # standard proxgrad
objective_nnscale, _, _ = nnmtf_test(:nnscale) # our rescaling trick
objective_simplexB, _, _ = nnmtf_test(:simplex; projectionA=:nonnegative) # proxgrad only enforce B is in simplex
objective_simplexA, _, _ = nnmtf_test(:simplex; projectionB=:nonnegative) # proxgrad only enforce A is in simplex

p = plot(objective_nnscale[2:end], yaxis=:log10, label="nnscale",
    xlabel="iteration", ylabel="relative error")
plot!(objective_simplex[2:end], yaxis=:log10, label="simplex A & B")
plot!(objective_simplexA[2:end], yaxis=:log10, label="simplex A only")
plot!(objective_simplexB[2:end], yaxis=:log10, label="simplex B only")
title!("size of Y is (I, J)=$((I, J))")
display(p)

end
