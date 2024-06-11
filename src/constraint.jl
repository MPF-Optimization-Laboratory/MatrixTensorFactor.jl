"""
Low level code for the types of constraints
"""

"""Abstract parent type for the verious constraints"""
abstract type AbstractConstraint <: Function end

"""
    GenericConstraint <: AbstractConstraint

General constraint. Simply applies apply and checks with check. Composing any two
`AbstractConstraint`s will return this type.

Calling a `GenericConstraint` on an `AbstractDecomposition` will use the function in the
field `apply`. Use `check(A, C::GenericConstraint)` to use the function in the field `check`.
"""
struct GenericConstraint <: AbstractConstraint
    apply::Function # input a AbstractDecomposition -> mutate it so that `check` would return true
    check::Function
end

function (C::GenericConstraint)(D::AbstractDecomposition)
    (C.apply)(D)
end

check(A::AbstractDecomposition, C::GenericConstraint) = (C.check)(A)

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

"""
struct ProjectedNormalization <: AbstractNormalization
    apply::Function # input a AbstractDecomposition -> mutate it so that `check` would return true
    check::Function # input a AbstractDecomposition -> output a Bool
end

struct ScaledNormalization{T<:Union{Real,AbstractArray{<:Real}}} <: AbstractNormalization
    norm::Function
    whats_normalized::Function
    scale::T
end

"""
    ScaledNormalization(norm; whats_normalized=identity, scale=1)

Main constructor for the constraint where `norm` of `whats_normalized` equals `scale`.

Scale can be a single `Real`, or an `AbstractArray{<:Real}`, but should be the same size as
the output of `whats_normalized`.
"""
ScaledNormalization(norm;whats_normalized=identity,scale=1)=ScaledNormalization{typeof(scale)}(norm, whats_normalized, scale)

function (S::ScaledNormalization)(A::AbstractDecomposition)
    whats_normalized_A = S.whats_normalized(A)
    A_norm = (S.norm).(whats_normalized_A) ./ S.scale(A)
    whats_normalized_A ./= A_norm
    return A_norm
end

check(A::AbstractDecomposition, S::ScaledNormalization) = all((S.norm).(S.whats_normalized(A)) .== S.scale)

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
    check(A::AbstractArray, C::EntryWise)::Bool

Checks if `A` is entrywise constrained
"""
check(A::AbstractArray, C::EntryWise) = all((C.check).(A))

const nnegative! = EntryWise(x -> max(0, x), x -> x >= 0)
