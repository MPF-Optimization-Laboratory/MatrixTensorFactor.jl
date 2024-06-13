"""
High level code for taking a tensor a performing a decomposition
"""

factorize(Y; kwargs...) =
	_factorize(Y; (default_kwargs(Y; kwargs...))...)

"""
Factor Y = ABC... such that
X is normalized according to normX for X in (A, B, C...)
"""
function _factorize(Y; kwargs...)
	decomposition = initialize_decomposition(Y; kwargs...)
	stats_data, getstats = initialize_stats(decomposition, Y; kwargs...)
	push!(stats_data, getstats(decomposition, Y; kwargs...))

	update! = make_update(decomposition, Y; kwargs...)
	@assert typeof(update!) <: AbstractUpdate{typeof(decomposition)} # basic check that the update is compatible with this decomposition

	while !converged(stats_data; kwargs...)
		# Update the decomposition
		# This is possibly one cycle of updates on each factor in the decomposition
		update!(decomposition)

		# Save stats
		push!(stats_data, getstats(decomposition, Y; kwargs...))
	end

	return decomposition, stats_data
end

function default_kwargs(Y; kwargs...)
	# Set up kwargs as a dictionary
	# then add `get!(kwargs, :option, default)` to set `option=default`
	# or use `do` syntax:
	# get!(kwargs, :option) do
	#     calculate_default()
	# end
	isempty(kwargs) ? kwargs = Dict{Symbol,Any}() : kwargs = Dict(kwargs)

	# Initialization
	get!(kwargs, :decomposition, nothing)
	get!(kwargs, :model, Tucker1)
	get!(kwargs, :rank, 1) # Can also be a tuple. For example, Tucker rank could be (1, 2, 3) for an order 3 array Y
	get!(kwargs, :init) do
		isnonnegative(Y) ? abs_randn : randn
	end
	# get!(kwargs, :freeze) # Default is handled by the model constructor

	# Update
	get!(kwargs, :algorithm, scaled_nn_block_gradient_decent)
	get!(kwargs, :core_constraint, l1normalize_1slices!)
	get!(kwargs, :whats_rescaled, (x -> eachcol(factor(x, 2))))

	# Stats
	get!(kwargs, :stats) do
		function grad_matrix(T::Tucker1)
			(C, A) = factors(T)
			CC = slicewise_dot(C, C)
			YC = slicewise_dot(Y, C)
			grad = A*CC - YC
			return grad
		end
		function grad_core(T::Tucker1)
			(C, A) = factors(T)
			AA = A'A
			YA = Y×₁A'
			grad = C×₁AA - YA
			return grad
		end
		(;
		gradient_norm = (X, Y, s) -> sqrt(norm(grad_matrix(X))^2 + norm(grad_core(X))^2),
		error    = (X, Y, s) -> norm(array(X) - Y),
		last_error_ratio = (X, Y, s) -> norm(array(X) - Y) / s[end, :error],
		)
	end

    return kwargs
end

function initialize_decomposition(Y; decomposition, model, rank, kwargs...)
	if !isnothing(decomposition)
		return decomposition
	else
		decomposition = model(size(Y), rank; kwargs...)
		return decomposition
	end
end

function make_update(decomposition, Y; algorithm, kwargs...)
	return algorithm(decomposition, Y; kwargs...)
end

function initialize_stats(decomposition, Y; stats, kwargs...)

	stats_data = DataFrame(make_columns(; kwargs...)...)

	function getstats(decomposition, Y; kwargs...)
		return
	end

	return stats_data, getstats
end
