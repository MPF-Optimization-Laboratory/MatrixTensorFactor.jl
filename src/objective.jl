"""
AbstractObjective <: Function

General interface is

struct L2 <: AbstractObjective end

after constructing

myobjective = L2()

you can call

myobjective(X, Y)
"""
abstract type AbstractObjective <: Function end

struct L2 <: AbstractObjective end

(O::L2)(X, Y) = norm2(X - Y)
