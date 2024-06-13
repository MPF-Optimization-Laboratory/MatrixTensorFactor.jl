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
	stats_data = initialize_stats(decomposition, Y; kwargs...)
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
