using Random
using KernelDensity
using Distributions
using Plots
using Pkg
Pkg.add("TensorOperations")
using TensorOperations

function generate_tensor(dims::Tuple{Int,Int,Int}, R::Int, continuous::Int = 1)
    if continuous == 1
        return generate_tensor_streams(dims, R)
    elseif continuous == 2
        return generate_tensor_2D_distributions(dims, R)
    else
        throw(ArgumentError("Invalid continuous value"))
    end
end


# Function to generate synthetic tensor with mixed distributions
function generate_tensor_streams(dims::Tuple{Int,Int,Int}, R::Int)
    I, J, K = dims[1], dims[2], dims[3]

    # Should I ensure that all sources are different from each other?
    # sources_matrix is an R x J matrix of normal distributions
    sources_matrix = [Normal(20*rand() - 10, 1 + 4*rand()) for _ in 1:R, _ in 1:J]

    # x values to sample the densities at
    start_val = -20
    end_val = 20
    x = range(start_val, end_val, length=K) 
    Δx = x[2] - x[1]

    α = ones(I)  # Need confirmation that this is fine
    mixing_matrix = rand(Dirichlet(α), R)  # Generate a I x R matrix

    # Normalize the proportion matrix so all rows sum to 1
    row_sums = sum(mixing_matrix, dims=2)
    mixing_matrix_normal = mixing_matrix ./ row_sums  
    

    # Make necessary tensor size
    Y_prime = zeros(Float64, (R, J, K))  # Pre-allocate tensor

    fixed_steps = start_val .+ (0:K-1) * Δx  # Create fixed steps based on the start value and step size

    # Fill the tensor with density values from the distributions at fixed steps
    for r in 1:R
        for j in 1:J
            Y_prime[r, j, :] .= pdf(sources_matrix[r, j], fixed_steps)
        end
    end
    Y_prime .*= Δx
    
    @tensor Y[i, j, k] := mixing_matrix_normal[i, r] * Y_prime[r, j, k]

    return (Y, mixing_matrix_normal, Y_prime)
end




function generate_tensor_2D_distributions(dims::Tuple{Int,Int,Int}, R::Int)
    I, J, K = dims[1], dims[2], dims[3]

end


Y, matrix, Y_prime = generate_tensor_streams((5, 3, 50), 4)



