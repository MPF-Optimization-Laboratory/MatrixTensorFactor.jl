"""
    MultiFactorize(Y; rank=1, model=Tucker1, kwargs...)

Perform a multi-stage factorization of the input the tensor `Y` following the tucker1 model. 

This function should be used when the data is continuous accross the third dimension of the tensor

# Arguments
- `Y::AbstractArray{T, 3}`: The input tensor to be factorized.
- See [`default_kwargs`](@ref) for the default keywords.

# Returns
A tuple `(decomposition, output_array, output_kwargs)`:
- `decomposition`: Tuple containing the core tensor and proportion matrix
- `stats::Vector`: Statistics of intermediate results collected during the iterative steps.
- `output_kwargs::Dict`: Additional outputs from the final `factorize` call.

# Example
dims = (10, 20, 128)
rank = 4
Y = generate_tensor_streams(dims, rank)  

decomposition, stats, kwargs = MultiFactorize(Y;
    rank=4,
    tolerence=(0.05),
    converged=(RelativeError),
    constraints=[simplex_12slices!, simplex_rows!],
    stats=[RelativeError])

"""
function MultiFactorize(Y; kwargs...)
    check_inputs(Y; kwargs...)
    final_size = size(Y, 3)
    decomposition = get(kwargs, :decomposition, nothing)
    stats = []
    indices = [1, div(final_size + 1, 2), final_size]

    while length(indices) < final_size
        Y_prime = Y[:, :, indices]
        scale_factor = final_size / size(Y_prime, 3)
        Y_prime .*= scale_factor
        decomposition, stats_data, output_kwargs = factorize(Y_prime; model=Tucker1, constrain_init=true, decomposition=decomposition, kwargs...)
        push!(stats, stats_data)
        
		indices, newly_added_indices = update_indices(indices)

        decomposition = resize_decomp_linear(decomposition, indices, newly_added_indices)
    end

    # Final factorization on the full tensor
    decomposition, stats_data, output_kwargs = factorize(Y; decomposition=decomposition, constrain_init=true, kwargs...)
    push!(stats, stats_data)
    
    return (decomposition, stats, output_kwargs)
end

function update_indices(indices)
    new_indices = []
    for i in 1:(length(indices) - 1)
        push!(new_indices, indices[i])
        push!(new_indices, div(indices[i] + indices[i + 1], 2))
    end
    push!(new_indices, indices[end])
    new_indices = unique(sort(new_indices))
    newly_added_indices = setdiff(new_indices, indices)

    return (new_indices, newly_added_indices)
end

function check_inputs(Y; model=nothing, max_iter=nothing, kwargs...)
    if !isa(Y, AbstractArray)
        throw(ArgumentError("Input tensor Y is not an AbstractArray"))
    end
    if !(ndims(Y) == 3)
        num = ndims(Y)
        throw(ArgumentError("Dimension of Y must be 3; currently is $num"))
    end
    if model != Tucker1 && !isnothing(model) 
        throw(ArgumentError("Model must be Tucker1, currently is $model"))
    end
    if max_iter == Inf 
        @warn "maxiter not compatible for internal iterations. Defaulting last internal iterations to 1000"
    end
end

function resize_decomp_linear(decomp, indices, newly_added_indices)
    tensor = factors(decomp)[1]
    factor_matrix = factors(decomp)[2]
    new_tensor = Array{eltype(tensor)}(undef, size(tensor, 1), size(tensor, 2), length(indices))
    j = 1
    for (i, val) in enumerate(indices)
        if val in newly_added_indices
            new_tensor[:, :, i] = (tensor[:, :, j-1] + tensor[:, :, j]) / 2
        else
            new_tensor[:, :, i] = tensor[:, :, j]
            j += 1  
        end
    end

    new_tensor .*= ((length(indices) - length(newly_added_indices)) / length(indices))
    
    return Tucker1((new_tensor, factor_matrix))
end
