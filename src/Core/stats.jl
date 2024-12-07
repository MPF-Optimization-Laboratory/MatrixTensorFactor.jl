"""
An AbstractStat is a type which, when created, can be applied to the four arguments
(X::AbstractDecomposition, Y::AbstractArray, previous::Vector{<:AbstractDecomposition}, parameters::Dict)
to (usually) return a number.
"""
abstract type AbstractStat <: Function end

# These methods allow us to check length([GradientNNCone, ObjectiveValue]) or length(GradientNNCone)
# or iterate over a list / single element of AbstractStat
Base.length(_::Type{<:AbstractStat}) = 1
Base.iterate(x::Type{<:AbstractStat}, state=1) = state > 1 ? nothing : (x, state+1)
#Base.eltype(x::Type{<:AbstractStat}) = typeof(x)

struct Iteration <: AbstractStat
    function Iteration(; kwargs...) # must define it this way so the constructor can take (and ignore) kwargs
        new()
    end
end
# This does not work since the pattern (; kwargs...) counts as the same input as the empty ()
# so Julia thinks you are redefining Iteration() = Iteration() in a circular manner
# Iteration(; kwargs...) = Iteration()

struct GradientNorm{T} <: AbstractStat
    gradients::T
    function GradientNorm{T}(gradients) where T
        @assert eltype(gradients) <: Function
        new{T}(gradients)
    end
end
GradientNorm(; gradients, kwargs...) = GradientNorm{typeof(gradients)}(gradients)

struct GradientNNCone{T} <: AbstractStat
    gradients::T
    function GradientNNCone{T}(gradients) where T
        @assert eltype(gradients) <: Function
        new{T}(gradients)
    end
end
GradientNNCone(; gradients, kwargs...) = GradientNNCone{typeof(gradients)}(gradients)

struct ObjectiveValue{T<:AbstractObjective} <: AbstractStat
    objective::T
end
ObjectiveValue(; objective, kwargs...) = ObjectiveValue(objective)

struct ObjectiveRatio{T<:AbstractObjective} <: AbstractStat
    objective::T
end
ObjectiveRatio(; objective, kwargs...) = ObjectiveRatio(objective)

struct RelativeError{T<:Function} <: AbstractStat
    norm::T
end
RelativeError(; norm, kwargs...) = RelativeError(norm)

struct IterateNormDiff{T<:Function} <: AbstractStat
    norm::T
end

IterateNormDiff(; norm, kwargs...) = IterateNormDiff(norm)

struct IterateRelativeDiff{T<:Function} <: AbstractStat
    norm::T
end

IterateRelativeDiff(; norm, kwargs...) = IterateRelativeDiff(norm)

"""
    PrintStats(; kwargs...)

Does not use any of the kwargs. Simply prints the most recent row of the stats.
"""
struct PrintStats <: AbstractStat
    function PrintStats(; kwargs...) # must define it this way so the constructor can take (and ignore) kwargs
        new()
    end
end

"""
    DisplayDecomposition(; kwargs...)

Does not use any of the kwargs. Simply displays the current iteration.
"""
struct DisplayDecomposition <: AbstractStat
    function DisplayDecomposition(; kwargs...) # must define it this way so the constructor can take (and ignore) kwargs
        new()
    end
end

"""
The 2-norm of the stepsizes that would be taken for all blocks.

For example, if there are two blocks, and we would take a stepsize of A to update one block
and B to update the other, this would return sqrt(A^2 + B^2).
"""
struct EuclidianStepSize{T} <: AbstractStat
    steps::T
    function EuclidianStepSize{T}(steps) where T
        @assert all(x -> typeof(x) <: AbstractStep, steps)
        new{T}(steps)
    end
end

EuclidianStepSize(; steps, kwargs...) = EuclidianStepSize{typeof(steps)}(steps)

"""
The 2-norm of the lipshitz constants that would be taken for all blocks.

Need the stepsizes to be lipshitz steps since it is calculated similarly to EuclidianStepSize.
"""
struct EuclidianLipshitz{T} <: AbstractStat
    steps::T
    function EuclidianLipshitz{T}(steps) where T
        @assert all(x -> typeof(x) <: AbstractStep, steps)
        new{T}(steps)
    end
end

EuclidianLipshitz(; steps, kwargs...) = EuclidianLipshitz{typeof(steps)}(steps)

"""
    FactorNorms(; norm, kwargs...)

Makes a tuple containing the norm of each factor in the decomposition.
"""
struct FactorNorms{T<:Function} <: AbstractStat
    norm::T
end
FactorNorms(; norm, kwargs...) = FactorNorms(norm)

function (S::Iteration)(_, _, _, parameters, stats)
    @assert nrow(stats) == parameters[:iteration] # make sure these don't drift for some reason
    return parameters[:iteration]
end
(S::GradientNorm)(X, _, _, _, _) = sqrt(sum(g -> norm2(g(X)), S.gradients))
function (S::GradientNNCone)(X, _, _, _, _)
    function d((A, g))
        grad = g(X)
        return norm2(@view grad[@. (A > 0) | (grad < 0)]) # faster to use full "or" rather than shortcutting "or" in this case
    end
    return sqrt(sum(d, zip(factors(X), S.gradients)))
end
(S::ObjectiveValue)(X, Y, _, _, _) = S.objective(X, Y)
(S::ObjectiveRatio)(X, Y, previous, _, _) = S.objective(previous[begin], Y) / S.objective(X, Y) # converged if < 1.01 say
# TODO compute less with this, but need to ensure stats
# are calculated in the right order, and dependent stats are calculated:
# stats[end-1, :ObjectiveValue] / stats[end, :ObjectiveValue]
(S::RelativeError)(X, Y, _, _, _) = S.norm(X - Y) / S.norm(Y)

(S::IterateNormDiff)(X, _, previous, _, _) = S.norm(X - previous[begin])
(S::IterateRelativeDiff)(X, _, previous, _, _) = S.norm(X - previous[begin]) / S.norm(previous[begin])
(S::EuclidianStepSize)(X, _, _, _, _) = sqrt.(sum(calcstep -> calcstep(X)^2, S.steps))
(S::EuclidianLipshitz)(X, _, _, _, _) = sqrt.(sum(calcstep -> calcstep(X)^(-2), S.steps))
(S::FactorNorms)(X, _, _, _, _) = S.norm.(factors(X))
(S::PrintStats)(_, _, _, parameters, stats) = if parameters[:iteration] > 0; println(last(stats)); end
function (S::DisplayDecomposition)(X, _, _, parameters, _)
    println("iteration ", parameters[:iteration])
    display(X)
    return nothing
end
