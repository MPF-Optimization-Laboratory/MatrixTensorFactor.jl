using Random
using KernelDensity
using Distributions
using Plots
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

"""
    generate_tensor_streams(dims::Tuple{Int,Int,Int}, R::Int) -> Tuple{Array{Float64, 3}, Array{Float64, 2}, Array{Float64, 3}}

Generate a 3D tensor that is coninuous accross the third dimension and all values in sum(Y, dims=3) are 1.

This function creates a synthetic tensor `Y` by computing the mode-1 product from a proportion matrix and core tensor. 
    The proportion matrix is created by a Dirichlet matrix.
    The core tensor has density values sampled from a normal distribution

# Arguments
- `dims::Tuple{Int,Int,Int}`: A tuple `(I, J, K)` specifying the dimensions of the tensor to be generated:
  - `I`: Number of streams (rows in the mixing matrix).
  - `J`: Number of distributions per stream.
  - `K`: Number of density sampling points.
- `R::Int`: The number of source distributions used for mixing.

# Returns
A tuple containing:
1. `Y::Array{Float64, 3}`: The generated synthetic tensor of size `(I, J, K)`.
2. `mixing_matrix::Array{Float64, 2}`: The mixing matrix of size `(I, R)`, where each row sums to 1.
3. `core_tensor::Array{Float64, 3}`: The core tensor of size `(R, J, K)` where all values in sum(core_tensor, dims=3) are 1.


# Examples
dims = (10, 5, 100)  # Tensor dimensions: 10x5x100
R = 3                # Number of source distributions

Y, mixing_matrix, core_tensor = generate_tensor_streams(dims, R)
"""
function generate_tensor_streams(dims::Tuple{Int,Int,Int}, R::Int)
    I, J, K = dims[1], dims[2], dims[3]

    # Should I ensure that all sources are different from each other?
    sources_matrix = [Normal(20*rand() - 10, 1 + 4*rand()) for _ in 1:R, _ in 1:J]

    # x values to sample the densities at
    start_val = -20
    end_val = 20
    x = range(start_val, end_val, length=K) 
    Δx = x[2] - x[1]

    α = ones(I)  # Need confirmation that this is fine
    mixing_matrix = rand(Dirichlet(α), R) 

    row_sums = sum(mixing_matrix, dims=2)
    mixing_matrix ./= row_sums  
    
    core_tensor = zeros(Float64, (R, J, K)) 

    fixed_steps = start_val .+ (0:K-1) * Δx  

    for r in 1:R
        for j in 1:J
            core_tensor[r, j, :] .= pdf(sources_matrix[r, j], fixed_steps)
        end
    end
    core_tensor .*= Δx
    
    @tensor Y[i, j, k] := mixing_matrix[i, r] * core_tensor[r, j, k]

    return (Y, mixing_matrix, core_tensor)
end




function generate_tensor_2D_distributions(dims::Tuple{Int,Int,Int}, R::Int)
    # TODO: Implement data that is continuous accross 2 dimensional slices
    I, J, K = dims[1], dims[2], dims[3]

end

"""
    normalize_fibers!(array::AbstractArray{<:Real, 3}) -> AbstractArray{<:Real, 3}

Normalize the fibers of a 3D array in place.

This function normalizes the "fibers" of a 3-dimensional array along the third dimension, ensuring that the sum of each fiber is equal to 1. If the sum of a fiber is zero, it remains unchanged.

# Returns
- The same input array, modified in place, with each fiber normalized.
"""
function normalize_fibers!(array::AbstractArray{<:Real, 3})
    I, J, K = size(array)
    for i in 1:I
        for j in 1:J
            fiber_sum = sum(array[i, j, :])
            if fiber_sum != 0
                array[i, j, :] .= array[i, j, :] ./ fiber_sum
            end
        end
    end
    return array
end



Y, matrix, Y_prime = generate_tensor_streams((5, 3, 50), 4)



