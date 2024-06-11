"""
Low level code for the types of constraints
"""

"""Abstract parent type for the verious constraints"""
abstract type AbstractConstraint <: Function end

"""
    check(C::AbstractConstraint, A::AbstractArray)::Bool

Returns `true` if `A` satisfies the constraint `C`.
"""
function check(_::AbstractConstraint, _::AbstractArray); end

"""
    GenericConstraint <: AbstractConstraint

General constraint. Simply applies apply and checks with check. Composing any two
`AbstractConstraint`s will return this type.

Calling a `GenericConstraint` on an `AbstractArray` will use the function in the
field `apply`. Use `check(C::GenericConstraint, A)` to use the function in the field `check`.
"""
struct GenericConstraint <: AbstractConstraint
    apply::Function # input a AbstractArray -> mutate it so that `check` would return true
    check::Function
end

function (C::GenericConstraint)(D::AbstractArray)
    (C.apply)(D)
end

check(C::GenericConstraint, A::AbstractArray) = (C.check)(A)

# TODO should I be overloading the composition function like this?
∘(f::AbstractConstraint, g::AbstractConstraint) = GenericConstraint(f.apply ∘ g.apply, bool_function_and(f.check, g.check))
∘(f::AbstractConstraint, g::Function) = f.apply ∘ g
∘(f::Function, g::AbstractConstraint) = f ∘ g.apply

struct BoolFunctionAnd <: Function
    f::Function
    g::Function
end

(F::BoolFunctionAnd)(x) = F.f(x) & F.g(x) # No short curcit to ensure any warnings are shown from both checks

bool_function_and(f::Function, g::Function) = BoolFunctionAnd(f, g)

abstract type AbstractNormalization <: AbstractConstraint end

"""
    ProjectedNormalization(projection, norm; whats_normalized=identity)

Main constructor for the constraint where `norm` of `whats_normalized` equals `scale`.

Scale can be a single `Real`, or an `AbstractArray{<:Real}`, but should be the same size as
the output of `whats_normalized`.
"""
struct ProjectedNormalization <: AbstractNormalization
    norm::Function # input a AbstractArray -> output a Bool
    projection::Function # input a AbstractArray -> mutate it so that `check` would return true
    whats_normalized::Function
end

ProjectedNormalization(norm, projection; whats_normalized=identity)=ProjectedNormalization(norm, projection, whats_normalized)

function (P::ProjectedNormalization)(A::AbstractArray)
    whats_normalized_A = P.whats_normalized(A)
    (P.projection).(whats_normalized_A)
end

check(P::ProjectedNormalization, A::AbstractArray) = all((P.norm).(P.whats_normalized(A)) .== 1)

"""
    ScaledNormalization(norm; whats_normalized=identity, scale=1)

Main constructor for the constraint where `norm` of `whats_normalized` equals `scale`.

Scale can be a single `Real`, or an `AbstractArray{<:Real}`, but should be the same size as
the output of `whats_normalized`.
"""
struct ScaledNormalization{T<:Union{Real,AbstractArray{<:Real}}} <: AbstractNormalization
    norm::Function
    whats_normalized::Function
    scale::T
end

ScaledNormalization(norm;whats_normalized=identity,scale=1)=ScaledNormalization{typeof(scale)}(norm, whats_normalized, scale)

function (S::ScaledNormalization)(A::AbstractArray)
    whats_normalized_A = S.whats_normalized(A)
    A_norm = (S.norm).(whats_normalized_A) ./ S.scale(A)
    whats_normalized_A ./= A_norm
    return A_norm
end

check(S::ScaledNormalization, A::AbstractArray) = all((S.norm).(S.whats_normalized(A)) .== S.scale)

"""Entrywise constraint. Note both apply and check needs to be performed entrywise on an array"""
struct EntryWise <: AbstractConstraint
    apply::Function
    check::Function
end

"""Make entrywise callable, by applying the constraint entrywise to arrays"""
function (C::EntryWise)(A::AbstractArray)
    A .= (C.apply).(A)
end

"""
    check(C::EntryWise, A::AbstractArray)::Bool

Checks if `A` is entrywise constrained
"""
check(C::EntryWise, A::AbstractArray) = all((C.check).(A))

const nnegative! = EntryWise(x -> max(0, x), x -> x >= 0)
