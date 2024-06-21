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

ProjectedNormalization(norm, projection; whats_normalized=identityslice)=ProjectedNormalization(norm, projection, whats_normalized)

function (P::ProjectedNormalization)(A::AbstractArray)
    whats_normalized_A = P.whats_normalized(A)
    (P.projection).(whats_normalized_A)
end

check(P::ProjectedNormalization, A::AbstractArray) = all((P.norm).(P.whats_normalized(A)) .≈ 1)

### Some standard projections ###
l2norm(x::AbstractArray) = sqrt(norm2(x))
function l2project!(x::AbstractArray)
    if iszero(x)
        @warn "Input $x is zero, picking a closest element"
        x .= ones(size(x)) ./ sqrt(length(A))
        return
    end

    x ./= l2norm(x)
end

const l2normalize! = ProjectedNormalization(l2norm, l2project!)
const l2normalize_rows! = ProjectedNormalization(l2norm, l2project!; whats_normalized=eachrow)
const l2normalize_cols! = ProjectedNormalization(l2norm, l2project!; whats_normalized=eachcol)
const l2normalize_1slices! = ProjectedNormalization(l2norm, l2project!; whats_normalized=(x -> eachslice(x; dims=1)))
const l2normalize_12slices! = ProjectedNormalization(l2norm, l2project!; whats_normalized=(x -> eachslice(x; dims=(1,2))))

l1norm(x::AbstractArray) = mapreduce(abs, +, x)
function l1project!(x::AbstractArray)
    if iszero(x)
        @warn "Input $x is zero, picking a closest element"
        x .= ones(size(x)) ./ length(A)
        return
    end

    signs = sign.(x)
    x .= signs .* projsplx(signs .* x)
end

const l1normalize! = ProjectedNormalization(l1norm, l1project!)
const l1normalize_rows! = ProjectedNormalization(l1norm, l1project!; whats_normalized=eachrow)
const l1normalize_cols! = ProjectedNormalization(l1norm, l1project!; whats_normalized=eachcol)
const l1normalize_1slices! = ProjectedNormalization(l1norm, l1project!; whats_normalized=(x -> eachslice(x; dims=1)))
const l1normalize_12slices! = ProjectedNormalization(l1norm, l1project!; whats_normalized=(x -> eachslice(x; dims=(1,2))))

linftynorm(x::AbstractArray) = maximum(abs, x)
function linftyproject!(x::AbstractArray)
    if iszero(x)
        @warn "Input $x is zero, picking a closest element"
        x .= zero(x)
        x[begin] = 1
        return
    end

    xnorm = linftynorm(x)
    if xnorm > 1
        # projection to the l_infinity ball
        x .= clamp.(x, -1, 1)
    elseif xnorm < 1
        # push the closest element to ±1 to sign(x[i])
        indexes = findall(xi -> abs(xi) == xnorm, x)
        length(indexes) == 1 ||
            @warn "L_infinity projection is not unique, picking a closest element"
        i = indexes[begin]
        x[i] = sign(x[i])
    # else, we have xnorm == 1 already so do nothing
    end
end

const linftynormalize! = ProjectedNormalization(linftynorm, linftyproject!)
const linftynormalize_rows! = ProjectedNormalization(linftynorm, linftyproject!; whats_normalized=eachrow)
const linftynormalize_cols! = ProjectedNormalization(linftynorm, linftyproject!; whats_normalized=eachcol)
const linftynormalize_inftyslices! = ProjectedNormalization(linftynorm, linftyproject!; whats_normalized=(x -> eachslice(x; dims=1)))
const linftynormalize_12slices! = ProjectedNormalization(linftynorm, linftyproject!; whats_normalized=(x -> eachslice(x; dims=(1,2))))

"""
    ScaledNormalization(norm; whats_normalized=identity, scale=1)

Main constructor for the constraint where `norm` of `whats_normalized` equals `scale`.

Scale can be a single `Real`, or an `AbstractArray{<:Real}`, but should be brodcast-able
with the output of `whats_normalized`.
Lasly, scale can be a `Function` which will act on an `AbstractArray{<:Real}` and return
something that is brodcast-able `whats_normalized`.
"""
struct ScaledNormalization{T<:Union{Real,AbstractArray{<:Real},Function}} <: AbstractNormalization
    norm::Function
    whats_normalized::Function
    scale::T
end

ScaledNormalization(norm;whats_normalized=identityslice,scale=1) = ScaledNormalization{typeof(scale)}(norm, whats_normalized, scale)

function (S::ScaledNormalization{T})(A::AbstractArray) where {T<:Union{Real,AbstractArray{<:Real}}}
    whats_normalized_A = S.whats_normalized(A)
    A_norm = (S.norm).(whats_normalized_A) ./ S.scale
    whats_normalized_A ./= A_norm
    return A_norm
end

function (S::ScaledNormalization{T})(A::AbstractArray) where {T<:Function}
    whats_normalized_A = S.whats_normalized(A)
    A_norm = (S.norm).(whats_normalized_A) ./ S.scale(A)
    whats_normalized_A ./= A_norm
    return A_norm
end

check(S::ScaledNormalization, A::AbstractArray) = all((S.norm).(S.whats_normalized(A)) .≈ S.scale)

### Some standard rescaling ###

const l2scaled! = ScaledNormalization(l2norm)
const l2scaled_rows! = ScaledNormalization(l2norm; whats_normalized=eachrow)
const l2scaled_cols! = ScaledNormalization(l2norm; whats_normalized=eachcol)
const l2scaled_1slices! = ScaledNormalization(l2norm; whats_normalized=(x -> eachslice(x; dims=1)))
const l2scaled_12slices! = ScaledNormalization(l2norm; whats_normalized=(x -> eachslice(x; dims=(1,2))))

const l1scaled! = ScaledNormalization(l1norm)
const l1scaled_rows! = ScaledNormalization(l1norm; whats_normalized=eachrow)
const l1scaled_cols! = ScaledNormalization(l1norm; whats_normalized=eachcol)
const l1scaled_1slices! = ScaledNormalization(l1norm; whats_normalized=(x -> eachslice(x; dims=1)))
const l1scaled_12slices! = ScaledNormalization(l1norm; whats_normalized=(x -> eachslice(x; dims=(1,2))))

const l1scaled_average12slices! = ScaledNormalization(l1norm;
    whats_normalized=(x -> eachslice(x; dims=1)),
    scale=(A -> size(A)[2])) # the length of the second dimention "J"

const linftyscaled! = ScaledNormalization(linftynorm)
const linftyscaled_rows! = ScaledNormalization(linftynorm; whats_normalized=eachrow)
const linftyscaled_cols! = ScaledNormalization(linftynorm; whats_normalized=eachcol)
const linftyscaled_1slices! = ScaledNormalization(linftynorm; whats_normalized=(x -> eachslice(x; dims=1)))
const linftyscaled_12slices! = ScaledNormalization(linftynorm; whats_normalized=(x -> eachslice(x; dims=(1,2))))

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
