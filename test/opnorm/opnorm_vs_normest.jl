using Random: randn
using LinearAlgebra #: opnorm, Symmetric
using BenchmarkTools

#using LinearOperators: normest

"""
  normest(S) estimates the matrix 2-norm of S.
  This function is an adaptation of Matlab's built-in NORMEST.
  This method allocates.

  -----------------------------------------
  Inputs:
    S --- Matrix or LinearOperator type,
    tol ---  relative error tol, default(or -1) Machine eps
    maxiter --- maximum iteration, default 100

  Returns:
    e --- the estimated norm
    cnt --- the number of iterations used
  """
function normest(S, tol = -1, maxiter = 100)
  (m, n) = size(S)
  cnt = 0
  if tol == -1
    tol = Float64(eps(eltype(S)))
  end
  # Compute an "estimate" of the ab-val column sums.
  v = ones(eltype(S), m)
  v[randn(m) .< 0] .= -1
  x = zeros(eltype(S), n)
  mul!(x, S', v)
  e = norm(x)

  if e == 0
    return e, cnt
  end

  x ./= e
  e_0 = zero(e)
  Sx = zeros(eltype(S), m) # fixed from Sx = zeros(eltype(S), n)

  while abs(e - e_0) > tol * e
    e_0 = e
    mul!(Sx, S, x)
    if count(x -> x != 0, Sx) == 0
      Sx .= randn(eltype(Sx), size(Sx))
    end
    mul!(x, S', Sx)
    normx = norm(x)
    e = normx / norm(Sx)
    x ./= normx
    cnt = cnt + 1
    if cnt > maxiter
      @warn("normest did not converge ", maxiter, tol,)
      break
    end
  end

  return e, cnt
end

A = randn(100,10)

fopnorm1(X) = opnorm(X)
fopnorm2(X) = sqrt(opnorm(X'X))
fopnorm3(X) = sqrt(opnorm(Symmetric(X'X)))

mynormest(X) = normest(X, -1, 100000)[1] # just extract the estimated value

fnormest1(X) = mynormest(X)
fnormest2(X) = sqrt(mynormest(X'X))
fnormest3(X) = sqrt(mynormest(Symmetric(X'X)))

dimensions = (10000, 10)

b = @benchmark fopnorm1(X) setup=(X=randn(dimensions))
display(b)
b = @benchmark fopnorm2(X) setup=(X=randn(dimensions))
display(b)
b = @benchmark fopnorm3(X) setup=(X=randn(dimensions))
display(b)




b = @benchmark fnormest1(X) setup=(X=randn(dimensions))
display(b)
b = @benchmark fnormest2(X) setup=(X=randn(dimensions))
display(b)
b = @benchmark fnormest3(X) setup=(X=randn(dimensions))
display(b)










dimensions = (100, 100)

b = @benchmark fopnorm1(X) setup=(X=randn(dimensions))
display(b)
b = @benchmark fopnorm2(X) setup=(X=randn(dimensions))
display(b)
b = @benchmark fopnorm3(X) setup=(X=randn(dimensions))
display(b)
