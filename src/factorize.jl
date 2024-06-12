"""
High level code for taking a tensor a performing a decomposition
"""

general_tensor_factor(Y; kwargs...) =
	_general_tensor_factor(Y; (default_kwargs(; kwargs...))...)

"""
Factor Y = ABC... such that
X is normalized according to normX for X in (A, B, C...)
"""
function _general_tensor_factor(Y; kwargs...)
	decomposition, stats_data = initialize(kwargs...)
	push!(stats_data, stats(decomposition, Y; kwargs...))

	update! = make_update(decomposition, Y; kwargs...)
	@assert typeof(update!) <: AbstractUpdate{typeof(decomposition)} # basic check that the update is compatible with this decomposition

	while !converged(stats_data; kwargs...)
		# Update the decomposition
		# This is possibly one cycle of updates on each factor in the decomposition
		update!(decomposition)

		# Save stats
		push!(stats_data, stats(decomposition, Y; kwargs...))
	end

	return decomposition, stats_data
end

function default_kwargs(; kwargs...)
    return kwargs
end

function make_update(decomposition, Y; kwargs...)
	return block_gradient_decent
end
