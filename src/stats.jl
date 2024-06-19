"""
An AbstractStat is a type which, when created, can be applied to the four arguments
(X::AbstractDecomposition, Y::AbstractArray, previous::Vector{<:AbstractDecomposition}, parameters::Dict)
to return a number.
"""
abstract type AbstractStat <: Function end

struct Iteration <: AbstractStat end

struct GradientNorm{T<:AbstractVector{<:Function}} <: AbstractStat
    gradients::T
end

struct GradientNNCone{T<:AbstractVector{<:Function}} <: AbstractStat
    gradients::T
end

struct ObjectiveValue{T<:AbstractObjective} <: AbstractStat
    objective::T
end

struct ObjectiveRatio{T<:AbstractObjective} <: AbstractStat
    objective::T
end

#(S::Iteration)(_, _, _, stats) = nrow(stats) + 1
(S::GradientNorm)(X, _, _, _) = sqrt(mapreduce(g -> norm2(g(X)), +, S.gradients))
function (S::GradientNNCone)(X, _, _, _)
    function d(A, g)
        grad = g(X)
        return norm2(@view grad[@. (A > 0) | (grad < 0)])
    end
    return sqrt(mapreduce(d, +, zip(factors(X), S.gradients)))
end
(S::ObjectiveValue)(X, Y, _, _) = S.objective(X, Y)
(S::ObjectiveRatio)(X, Y, previous, _) = S.objective(previous[begin], Y) / S.objective(X, Y) # converged if < 1.01 say
# TODO compute less with this, but need to ensure stats
# are calculated in the right order, and dependent stats are calculated:
# stats[end-1, :ObjectiveValue] / stats[end, :ObjectiveValue]

(S::IterateNormDiff)(X, _, previous, _) = S.norm(X - previous[begin])
(S::IterateRelativeDiff)(X, _, previous, _) = S.norm(X - previous[begin]) / S.norm(previous[begin])
