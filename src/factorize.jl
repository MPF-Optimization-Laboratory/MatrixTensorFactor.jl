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
	previous, updateprevious! = initialize_previous(decomposition, Y; kwargs...)
	parameters, updateparameters! = initialize_parameters(decomposition, Y, previous; kwargs...)

	update!, kwargs = make_update(decomposition, Y; kwargs...) # more kwargs get added here, so need to return kwargs

	stats_data, getstats = initialize_stats(decomposition, Y, previous, parameters; kwargs...)

	converged = make_converged(; kwargs...)

	while !converged(stats_data; kwargs...)
		# Update the decomposition
		# This is possibly one cycle of updates on each factor in the decomposition
		update!(decomposition; parameters...)

		# Update parameters used for the next update of the decomposition
		# this could be the next stepsize or other info used by update!
		updateparameters!(parameters, decomposition, previous)

		# Save stats
		push!(stats_data, getstats(decomposition, Y, previous, parameters, stats_data))

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
	isempty(kwargs) ? kwargs = Dict{Symbol,Any}() : kwargs = Dict{Symbol,Any}(kwargs)

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
	get!(kwargs, :objective, L2())
	#get!(kwargs, :core_constraint, l1normalize_1slices!)
	#get!(kwargs, :whats_rescaled, (x -> eachcol(factor(x, 2))))
	# get!(kwargs, :random_order) # This default is handled by the BlockedUpdate struct

	# Momentum
	get!(kwargs, :momentum, true)
	get!(kwargs, :δ, 0.9999)
	get!(kwargs, :previous_iterates, 1)

	# Stats
	get!(kwargs, :stats, [Iteration, GradientNNCone, ObjectiveValue])
	if Iteration ∉ kwargs[:stats]
		kwargs[:stats] = [Iteration, kwargs[:stats]...] # not using pushfirst! since kwargs[:stats] could be a Tuple
	end
	@assert all(s -> s <: AbstractStat, kwargs[:stats])

	# Convergence
	get!(kwargs, :converged, GradientNNCone) # can be a single AbstractStat or a tuple/vector of them
											 # and must be a subset of kwargs[:stats]
	@assert all(s -> s <: AbstractStat, kwargs[:converged])
	get!(kwargs, :tolerence, 1e-6) # need one tolerence per stat
	@assert length(kwargs[:tolerence]) == length(kwargs[:converged])
	get!(kwargs, :maxiter, 300) # Iteration
	@assert all(c -> c in kwargs[:stats], kwargs[:converged]) # more memory efficient that all(in.(kwargs[:converged], (kwargs[:stats],)))


	############ kwargs check #############
	# eltype(kwargs[:stats]) <: AbstractStat # not true unless they are constructed
	# notice Iteration vs Iteration()


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
function make_update(decomposition, Y; momentum, kwargs...)
	ns = eachfactorindex(decomposition)

	# TODO this looks messy converting the NamedTuple kwargs to a Dictionary so more keywords can be added
	kwargs = Dict{Symbol,Any}(kwargs)
	kwargs[:momentum] = momentum # have to add this keyword back since it was extracted by make_update

	kwargs[:gradients] = [make_gradient(decomposition, n, Y; kwargs...) for n in ns]
	kwargs[:steps] = [LipshitzStep(make_lipshitz(decomposition, n, Y; kwargs...)) for n in ns] # TODO avoid hard coded lipshitz step
	if momentum
		error("Momentum blocked updates not handled yet")
	else
		update! = BlockedUpdate((GradientDescent(n, g, s) for (n, g, s) in zip(ns, kwargs[:gradients], kwargs[:steps])
		)...)
	end
	return update!, kwargs
end

"""The stats that will be saved every iteration"""
function initialize_stats(decomposition, Y, previous, parameters; stats, kwargs...)
	stat_functions = [S(; kwargs...) for S in stats] # construct the AbstractStats

	getstats(decomposition, Y, previous, parameters, stats_data) =
		Tuple(f(decomposition, Y, previous, parameters, stats_data) for f in stat_functions)

	stats_data = DataFrame((Symbol(S) => [v] for (S,v) in zip(stats, getstats(decomposition, Y, previous, parameters, DataFrame())))...)

	return stats_data, getstats
end

"""Keep track of one or more previous iterates"""
function initialize_previous(decomposition, Y; previous_iterates::Integer, kwargs...)
	previous = [deepcopy(decomposition) for _ in 1:previous_iterates] # TODO check if this should be copy?
	if previous_iterates == 0
		# No need to copy previous, so make a function that does nothing
		updateprevious1!(x...) = nothing
		return previous, updateprevious1!
	else
		# Shift the list of previous iterates and put the newest first
		function updateprevious2!(previous, parameters, decomposition)
			previous .= circshift(previous, 1)
			previous[begin] = deepcopy(decomposition) # TODO check if this should be copy, or copy!
			parameters[:x_last] = previous[begin]
		end
		return previous, updateprevious2!
	end
end

"""update parameters needed for the update"""
function initialize_parameters(decomposition, Y, previous; momentum::Bool, kwargs...)
	# parameters for the update step are symbol => value pairs
	# they are held in a dictionary since we may mutate these for ex. the stepsize
	parameters = Dict{Symbol, Any}()

	# General Looping
	parameters[:iteration] = 0
	parameters[:x_last] = previous[begin] # Last iterate

	# Momentum
	if momentum
		parameters[:t_last] = float(1) # need this field to hold Floats, not Ints
		parameters[:t] = update_t(float(1))
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

function make_converged(; converged, tolerence, maxiter, kwargs...)
	converged_tol = zip(converged, tolerence)
	function isconverged(stats_data; kwargs...)
		if any(((c, t),) -> stats_data[end, Symbol(c)] < t, converged_tol) # note the use of ((c, t),) rather than (c, t) since each element of the zipped converged_tol is a tuple, not two values
			if length(converged) == 1
				@info "converged based on $converged less than $tolerence"
			else
				indexes = findall(((c, t),) -> stats_data[end, Symbol(c)] < t, converged_tol)
				@info "converged based on $(join(converged[indexes], ", ", " and ")) less than $(join(tolerence[indexes], ", ", " and "))"
			end
			return true
		elseif stats_data[end, :Iteration] >= maxiter
			@warn "maximum iteration $maxiter reached, without convergence"
			return true
		else
			return false
		end
	end
	return isconverged
end
