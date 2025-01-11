# This file describes other MultiFactorize strategies that can possible be useful in certain occasions
# factorize, resize_decomp_linear and update_indices are defined in different files



# MultiFactorize where we log scale the internal iterations as another form of stopping the internal iterations
function MultiFactorize(Y; kwargs...)
    final_size = size(Y, 3)
    decomposition = get(kwargs, :decomposition, nothing)
    stats = []
    indices = [1, div(final_size + 1, 2), final_size]
    final_itr = get(kwargs, :maxiter, 1000)

    while length(indices) < final_size
        Y_prime = Y[:, :, indices]
        scale_factor = final_size / size(Y_prime, 3)
        Y_prime .*= scale_factor
        max_iter =  div(final_itr, scale_factor)
        decomposition, stats_data, output_kwargs = factorize(Y_prime; model=Tucker1, decomposition=decomposition, maxiter=max_iter, kwargs...)
        push!(stats, stats_data)
        
		indices, newly_added_indices = update_indices(indices)

        decomposition = resize_decomp_linear(decomposition, indices, newly_added_indices)
        max_iter = min(max_iter * 2, get(kwargs, :maxiter, Inf))
    end

    decomposition, stats_data, output_kwargs = factorize(Y; decomposition=decomposition, kwargs...)
    push!(stats, stats_data)
    
    return (decomposition, stats, output_kwargs)
end


# This version of MultiFactorize adds an extra constraint on the internal iterations
#   This internal constraint is to "converge" whenever we have flatlined our imporvement in RelativeError
#   This is done by adding a internal constraint of ObjectiveRatio where the value of 1.001 should be tuned
function MultiFactorize(Y; kwargs...)

    decomposition = get(kwargs, :decomposition, nothing)
    tolerence = (get(kwargs, :tolerence, () -> error("Missing :tolerence key"))..., ObjectiveRatio)
    constraints = (get(kwargs, :constraints, () -> error("Missing :constraints key"))..., 1.001)
    final_size = size(Y, 3)
    stats = []
    indices = [1, div(final_size + 1, 2), final_size]

    while length(indices) < final_size
        Y_prime = Y[:, :, indices]
        scale_factor = final_size / size(Y_prime, 3)
        Y_prime .*= scale_factor
        decomposition, stats_data, output_kwargs = factorize(Y_prime; model=Tucker1,tolerence=tolerence, constraints=constraints, decomposition=decomposition, kwargs...)
        push!(stats, stats_data)
        
		indices, newly_added_indices = update_indices(indices)

        decomposition = resize_decomp_linear(decomposition, indices, newly_added_indices)
    end

    decomposition, stats_data, output_kwargs = factorize(Y; decomposition=decomposition, kwargs...)
    push!(stats, stats_data)
    
    return (decomposition, stats, output_kwargs)
end