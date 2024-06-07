"""Code for taking a tensor a performing a decomposition"""

general_tensor_factor(Y; kwargs...) =
	_general_tensor_factor(Y; (default_kwargs(; kwargs...))...)


"""
Factor Y = ABC... such that
X is normalized according to normX for X in (A, B, C...)
"""
function _general_tensor_factor(Y; kwargs...)
	decomposition, stats_data = initialize(kwargs...)
	push!(stats_data, stats(Y, factors; kwargs...))

	while !converged(stats_data; kwargs...)
		for (A, X, Y) in eachfactor(decomposition)
			# Approx solve X in A*X = Y, where Y and A are fixed
			# This Y might be a reshaped/transposed version of the input to _general_tensor_factor
			update!(X; Y, A, kwargs...)
			constrain!(X, A; Y, kwargs...)
		end
		# This Y is now the original form
		push!(stats_data, stats(Y, decomposition; kwargs...))
	end

	return decomposition, stats_data
end

function default_kwargs(; kwargs...)
    nothing
end

function update!(X; Y, A, stepsize=stepsize(; kwargs...), kwargs...)

end
