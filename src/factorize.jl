"""
	factorize(Y; rank=1, model=Tucker1, kwargs...)

Factorizes `Y` according to the decomposition `model`.

See [`default_kwargs`](@ref) for the default keywords.
"""
factorize(Y; kwargs...) =
	_factorize(Y; (default_kwargs(Y; kwargs...))...)

"""
Inner level function once keyword agruments are set
"""
function _factorize(Y; kwargs...)
	decomposition, kwargs = initialize_decomposition(Y; kwargs...)
	previous, updateprevious! = initialize_previous(decomposition, Y; kwargs...)
	parameters, updateparameters! = initialize_parameters(decomposition, Y, previous; kwargs...)

	# one pass of the constraints is possibly applied so note decomposition could be mutated
	# and more kwargs get added here, so need to return kwargs
	update!, kwargs = make_update!(decomposition, Y; kwargs...)

	stats_data, getstats = initialize_stats(decomposition, Y, previous, parameters; kwargs...)

	converged = make_converged(; kwargs...)

	kwargs = NamedTuple(kwargs) # freeze the kwargs from a Dictionary to a NamedTuple for type stability

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

	return decomposition, stats_data, kwargs
end

"""
	default_kwargs(Y; kwargs...)

Handles all keywords and options, and sets defaults if not provided.

# Keywords & Defaults
## Initialization
- `decomposition`: `nothing`. Can provide a custom initialized AbstractDecomposition
- `model`: `Tucker1`, but overridden by the type of AbstractDecomposition if given `decomposition`
- `rank`: `1`, but overridden by the rank of AbstractDecomposition if given `decomposition`
- `init`: `abs_randn` for nonnegative inputs `Y`, `randn` otherwise
- `constrain_init`: `false`. Ensures the initalization satifies all given `constraints`
- `freeze`: the default frozen factors of the `model`

## Updates
- `objective`: `L2()`. Objective to minimize
- `norm`: `l2norm`. Norm to use for statistics, can be unrelated to the objective
- `random_order`: `false`. Perform the blocked updates in a random order each iteration

Momentum
- `momentum`: `true`
- `δ`: `0.9999`. Amount of momentum, between [0,1)
- `previous_iterates`: `1`. Number of pervious iterates to save and use between iterations

Constraints
- `constraints`: `nothing`. Can be a list of ConstraintUpdate, or just one

Stats
- `stats`: `[Iteration, ObjectiveValue, GradientNorm]` or in the case of nonnegative `Y`, `GradientNNCone` in place of `GradientNorm`
- `converged`: `GradientNorm` or in the case of nonnegative `Y`, `GradientNNCone`. What stat(s) to use for convergence. Will converge is any one of the provided stats is below their respective tolerence
- `tolerence`: `1`. A list the same length as `converged`
- `maxiter`: `300`. Additional stopping criterion if the number of iterations exceeds this number
"""
function default_kwargs(Y; kwargs...)
	# Set up kwargs as a dictionary
	# then add `get!(kwargs, :option, default)` to set `option=default`
	# or use `do` syntax:
	# get!(kwargs, :option) do
	#     calculate_default()
	# end
	isempty(kwargs) ? kwargs = Dict{Symbol,Any}() : kwargs = Dict{Symbol,Any}(kwargs)

	# Decomposition Initialization
	get!(kwargs, :decomposition, nothing)
	get!(kwargs, :model) do
		isnothing(kwargs[:decomposition]) ? Tucker1 : typeof(kwargs[:decomposition])
	end
	get!(kwargs, :rank) do # Can also be a tuple. For example, Tucker rank could be (1, 2, 3) for an order 3 array Y
		isnothing(kwargs[:decomposition]) ? 1 : rankof(kwargs[:decomposition])
	end
	get!(kwargs, :init) do
		isnonnegative(Y) ? abs_randn : randn
	end
	get!(kwargs, :constrain_init, false)
	# get!(kwargs, :freeze) # This default is handled by the model constructor
							# freeze=(...) can still be provided to override the default

	# Update
	get!(kwargs, :objective, L2()) # TODO handle arbitrary functions but constructing CustomObjective type
								   # this should call auto diff when making the gradients
	get!(kwargs, :norm, l2norm) # norm to use for data fitting analysis
	# note L2 <: AbstractObjective that takes two arguments L2()(X,Y) = norm2(X - Y)
	# whereas l2norm takes a single argument l2norm(X) = sqrt(norm2(X))
	# get!(kwargs, :random_order) # This default is handled by the BlockedUpdate struct

	# Momentum
	get!(kwargs, :momentum, true)
	get!(kwargs, :δ, 0.9999)
	get!(kwargs, :previous_iterates, 1)

	# Constraints
	get!(kwargs, :constraints, nothing)
	# the rest of the constraint parsing is handled later, once the decomposition is initalized

	# Stats
	get!(kwargs, :stats) do
		[Iteration, ObjectiveValue, isnonnegative(Y) ? GradientNNCone : GradientNorm]
	end
	if Iteration ∉ kwargs[:stats]
		kwargs[:stats] = [Iteration, kwargs[:stats]...] # not using pushfirst! since kwargs[:stats] could be a Tuple
	end
	@assert all(s -> s <: AbstractStat, kwargs[:stats])

	# Convergence
	get!(kwargs, :converged, isnonnegative(Y) ? GradientNNCone : GradientNorm) # can be a single AbstractStat or a tuple/vector of them
											 # and must be a subset of kwargs[:stats]
	@assert all(s -> s <: AbstractStat, kwargs[:converged])
	get!(kwargs, :tolerence, 1) # need one tolerence per stat
	@assert length(kwargs[:tolerence]) == length(kwargs[:converged])
	get!(kwargs, :maxiter, 300) # Iteration
	@assert all(c -> c in kwargs[:stats], kwargs[:converged]) # more memory efficient that all(in.(kwargs[:converged], (kwargs[:stats],)))

    return kwargs
end

"""
	parse_constraints(constraints, decomposition; kwargs...)

Parses the constraints to make sure we have a valid list of ConstraintUpdate
If only one AbstractConstraint is given, assume we want this constraint to apply to every
factor in the decomposition, and make a ConstraintUpdate for each factor.
"""
function parse_constraints(constraints, decomposition; kwargs...)
	# Base case
	if all(c -> typeof(c) <: ConstraintUpdate, constraints)
		return BlockedUpdate(constraints)
	else
		throw(error("got a set of constraints I am not sure how to parse: $constraints"))
	end
end
parse_constraints(constraints::Nothing, decomposition; kwargs...) = nothing

# Assume we want this constraint to apply to every factor in this case
parse_constraints(constraints::AbstractConstraint, decomposition; kwargs...) =
	BlockedUpdate([ConstraintUpdate(n, constraints; kwargs...) for n in eachfactorindex(decomposition)])

"""The decomposition model Y will be factored into"""
function initialize_decomposition(Y; decomposition, model, rank, kwargs...)
	kwargs = Dict{Symbol,Any}(kwargs)
	# have to add these keyword back since it was extracted by make_update
	kwargs[:model] = model
	kwargs[:rank] = rank

	if isnothing(decomposition)
		decomposition = model(size(Y), rank; kwargs...)
	end
	# have to add this keyword back since it was extracted by make_update
	kwargs[:decomposition] = decomposition
	return decomposition, kwargs
end

"""
What one iteration of the algorithm looks like.
One iteration is likely a full cycle through each block or factor of the model.
"""
function make_update!(decomposition, Y; momentum, constraints, constrain_init, kwargs...)
	ns = eachfactorindex(decomposition)

	kwargs = Dict{Symbol,Any}(kwargs)
	# have to add these keyword back since it was extracted by make_update
	kwargs[:momentum] = momentum
	kwargs[:constraints] = constraints
	kwargs[:constrain_init] = constrain_init

	kwargs[:gradients] = [make_gradient(decomposition, n, Y; kwargs...) for n in ns]
	kwargs[:steps] = [LipshitzStep(make_lipshitz(decomposition, n, Y; kwargs...)) for n in ns] # TODO avoid hard coded lipshitz step

	update! = BlockedUpdate((GradientDescent(n, g, s) for (n, g, s) in zip(ns, kwargs[:gradients], kwargs[:steps]))...)

	if momentum
		add_momentum!(update!)
	end

	if !isnothing(constraints)
		expanded_constraints! = parse_constraints(constraints, decomposition; kwargs...)
		kwargs[:constraints] = expanded_constraints! # save the expanded constraints
		smart_interlace!(update!, expanded_constraints!)
	end

	if constrain_init
		expanded_constraints!(decomposition)
		if !all(C -> check(C, decomposition), expanded_constraints!)
			indexes = findall(C -> !check(C, decomposition), expanded_constraints!)
			error("decomposition failed to be constrained. Check the constraint(s) $(expanded_constraints![indexes]) operation or the checking function")
		end
	end

	kwargs[:update] = update!

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
				indexes = findall(((c, t),) -> stats_data[end, Symbol(c)] < t, collect(converged_tol))
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
