"""
    rank_detect_factorize(Y; online_rank_estimation=false, rank=nothing, model=Tucker1, kwargs...)

Wraps `factorize()` with rank detection.

Selects the rank that maximizes the standard curvature of the Relative Error (as a function of rank).
"""
function rank_detect_factorize(Y; online_rank_estimation=false, rank=nothing, model=Tucker1, kwargs...)
    if isnothing(rank)
        # Initialize output and final error lists
        all_outputs = []
        final_rel_errors = Float64[]

        # Make sure RelativeError is part of the stats keyword argument
        kwargs = isempty(kwargs) ? Dict{Symbol,Any}() : Dict{Symbol,Any}(kwargs)
        get!(kwargs, :stats) do # If stats is not given, populate stats with RelativeError
            [Iteration, RelativeError, ObjectiveValue, isnonnegative(Y) ? GradientNNCone : GradientNorm]
        end
        if RelativeError ∉ kwargs[:stats] # If stats was given, make sure RelativeError is in the list stats
            kwargs[:stats] = [RelativeError, kwargs[:stats]...] # not using pushfirst! since kwargs[:stats] could be a Tuple
        end
        kwargs[:model] = model # add the model back into kwargs

        for rank in possible_ranks(Y, model)
            @info "Trying rank=$rank..."

            kwargs[:rank] = rank # add the rank into kwargs

            output = factorize(Y; kwargs...) # safe to call factorize (rather than _factorize) since both factorize and rank_detect_factorize have checks to see if the keyword `rank` is provided
            push!(all_outputs, output)
            _, stats, _ = output

            final_rel_error = stats[end, :RelativeError]
            push!(final_rel_errors, final_rel_error)
            @info "Final relative error = $final_rel_error"

            if (online_rank_estimation == true) && length(final_rel_errors) >= 3 # Need at least 3 points to evaluate curvature
                curvatures = standard_curvature(final_rel_errors)
                if curvatures[end] ≈ maximum(curvatures) # want the last curvature to be significantly smaller than the max
                    continue
                else
                    # we must have curvature[end] < maximum(curvature) so we can now return
                    R = argmax(curvatures)
                    @info "Optimal rank found: $R"
                    return ((all_outputs[R])..., final_rel_errors)
                end
            end
        end

        # Return if online_rank_estimation == false, or a clear rank was not found
        R = argmax(standard_curvature(final_rel_errors))
        @info "Optimal rank found: $R"
        return ((all_outputs[R])..., final_rel_errors)
    else
        return factorize(Y; rank, model, kwargs...)
    end
end

"""
    possible_ranks(Y, model)

Returns the rank of possible ranks `Y` could have under the `model`.

For matrices I×J this is 1:min(I, J). This is can be extended to tensors for different type
of decompositions.

Tucker-1 rank is ≤ min(I, prod(J1,...,JN)) for tensors I×J1×...×JN.

The CP-rank is ≤ minimum_{n} (prod(I1,...,IN) / In) for tensors I1×...×IN in general. Although
some shapes have have tighter upper bounds. For example, 2×I×I tensors over ℝ have a maximum
rank of floor(3I/2).
"""
function possible_ranks(Y, model)
    if model <: Tucker1
        I, Js... = size(Y)
        max_rank = min(I, prod(Js))
        return 1:max_rank
    elseif model <: CPDecomposition
        Is = size(Y)
        # There exist tighter upper bounds for particular shapes like I×I×K, but this a simple upper bound that works for all shapes
        max_rank = minimum(prod(Is) .÷ Is) # ÷ is Integer division
        return 1:max_rank
    else
        error("Possible ranks for models of type $model are not implemented")
    end
end
