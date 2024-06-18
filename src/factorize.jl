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
	previous, updateprevious! = initialize_previous(decomposition, Y; kwargs...)
	parameters, updateparameters! = initialize_parameters(decomposition, Y, previous; kwargs...)

	push!(stats_data, getstats(decomposition, Y, previous, parameters))

	update! = make_update(decomposition, Y; kwargs...)
	@assert typeof(update!) <: AbstractUpdate{typeof(decomposition)} # basic check that the update is compatible with this decomposition

	converged = make_converged(; kwargs...)

	while !converged(stats_data; kwargs...)
		# Update the decomposition
		# This is possibly one cycle of updates on each factor in the decomposition
		update!(decomposition; parameters...)

		# Update parameters used for the next update of the decomposition
		# this could be the next stepsize or other info used by update!
		updateparameters!(parameters, decomposition, previous)

		# Save stats
		push!(stats_data, getstats(decomposition, Y, previous, parameters))

		# Update the one or two previous iterates, used for momentum, spg stepsizes
		# We do not save every iterate since that is too much memory
		updateprevious!(previous, parameters, decomposition)
	end

	return decomposition, stats_data
end

"""Handles all keywords and options, and sets defaults if not provided"""
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
	# get!(kwargs, :freeze) # This default is handled by the model constructor
							# freeze=(...) can still be provided to override the default

	# Update
	get!(kwargs, :algorithm, scaled_nn_block_gradient_decent)
	get!(kwargs, :core_constraint, l1normalize_1slices!)
	get!(kwargs, :whats_rescaled, (x -> eachcol(factor(x, 2))))
	# get!(kwargs, :random_order) # This default is handled by the BlockedUpdate struct

	# Momentum
	get!(kwargs, :momentum, true)
	get!(kwargs, :δ, 0.9999)
	get!(kwargs, :previous_iterates, 1)

	# Stats
	get!(kwargs, :stats) do # TODO maybe make some types and structure for common stats you may want
							# some of these should match the algorithm, for example the gradient calculation
							# this is also not a viable way to store previous or the last two iterates
							# since this would need to save *every* iterate (too much memory)
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
		gradient_norm = (X, Y, stats) -> sqrt(norm(grad_matrix(X))^2 + norm(grad_core(X))^2),
		error    = (X, Y, stats) -> norm(array(X) - Y),
		last_error_ratio = (X, Y, stats) -> norm(array(X) - Y) / stats[end, :error],
		)
	end

    return kwargs
end

"""The decomposition model Y will be factored into"""
function initialize_decomposition(Y; decomposition, model, rank, kwargs...)
	if !isnothing(decomposition)
		return decomposition
	else
		decomposition = model(size(Y), rank; kwargs...)
		return decomposition
	end
end

"""
What one iteration of the algorithm looks like.
One iteration is likely a full cycle through each block or factor of the model.
"""
function make_update(decomposition, Y; algorithm, kwargs...)
	return algorithm(decomposition, Y; kwargs...)
end

"""The stats that will be saved every iteration"""
function initialize_stats(decomposition, Y; stats, kwargs...)

	stats_data = DataFrame(make_columns(; kwargs...)...)

	function getstats(decomposition, Y, previous, parameters)
		return
	end

	return stats_data, getstats
end

"""Keep track of one or more previous iterates"""
function initialize_previous(decomposition, Y; previous_iterates::Integer, kwargs...)
	previous = [deepcopy(decomposition) for _ in 1:previous_iterates] # TODO check if this should be copy?
	previous_iterates == 0 ? updateprevious!(x...) = nothing : begin
		# Shift the list of previous iterates and put the newest first
		function updateprevious!(previous, parameters, decomposition)
			previous .= circshift(previous, 1)
			previous[begin] = deepcopy(decomposition) # TODO check if this should be copy?
			parameters[:x_last] = previous[begin]
		end
	end
	return previous, updateprevious!
end

function initialize_parameters(decomposition, Y, previous; momentum::Bool, kwargs...)
	# parameters for the update step are symbol => value pairs
	# they are held in a dictionary since we may mutate these for ex. the stepsize
	parameters = Dict{Symbol, Any}()

	# General Looping
	parameters[:iteration] = 1
	parameters[:x_last] = previous[begin] # Last iterate

	# Momentum
	if momentum
		parameters[:t_last] = update_t(1::Float64)
		parameters[:t] = 1::Float64
		parameters[:ω] = (parameters[:t_last] - 1) / parameters[:t]
		parameters[:δ] = kwargs[:δ]
	end

	function updateparameters!(parameters, decomposition, previous)
		parameters[:iteration] += 1
		#parameters[:x_last] = previous[begin] # Last iterate updated with updateprevious!
											   # this ensures we only copy over the decomposition
											   # once since updateparameters! happens before updateprevious!

		# Momentum
		if momentum
			parameters[:t_last] = parameters[:t]
			parameters[:t] = update_t(parameters[:t_last])
			parameters[:ω] = (parameters[:t_last] - 1) / parameters[:t]
		end
	end

	return parameters, updateparameters!
end

update_t(t) = 0.5*(1 + sqrt(1 + 4*t^2))
