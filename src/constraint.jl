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

∘(f::AbstractConstraint, g::AbstractConstraint) = GenericConstraint(f.apply ∘ g.apply, f.check ∘ g.check)

"""

"""
struct Normalization <: AbstractConstraint
    apply::Function # input a AbstractDecomposition -> mutate it so that `check` would return true
    check::Function # input a AbstractDecomposition -> output a Bool
end


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
