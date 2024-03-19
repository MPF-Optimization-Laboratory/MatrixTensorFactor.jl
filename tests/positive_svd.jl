using LinearAlgebra
using Random

function proj!(x)
    x .= max.(0, x)
    x_norm = norm(x)
    if x_norm > 1
        x ./= x_norm
    end
end

"""Finds the best positive rank 1 approximation for H"""
function prank1(H; maxiter=10)
    x = rand(size(H)[2])
    step = 1 / opnorm(H)
    for _ in 1:maxiter
        x .+= H*x * step
        proj!(x)
        @show x'H*x
        @show x
    end
    return x, x'H*x
end

"""Finds the best rank 1 approximation for H"""
function rank1(H; maxiter=10)
    x = rand(size(H)[2])
    step = 1 / opnorm(H)
    for _ in 1:maxiter
        x .+= H*x * step
        normalize!(x)
        @show x'H*x
        @show x
    end
    return x, x'H*x
end

#######################################

M, R, N = 10, 5, 10

A = rand(M, R) / sqrt(M)
B = rand(R, N) / sqrt(N)

Y_true = A*B

H = Y_true'Y_true

x1, λp = prank1(H)

c1 = minimum(H ./ (x1*x1'))

H - λp * x1*x1'
H1 = H - c1 * x1*x1'

x2, λp = prank1(H1)

H1 - λp * x2*x2'

c2 = minimum(H1 ./ (x2*x2'))
H2 = H1 - c2 * x2*x2'


##############
"""
    curvilinear_bb(F, dF, X0; τ=1e-2, ρ1=1e-4, ρ2=sqrt(ρ1), δ=0.5, η=0.5, ϵ=1e-3, maxiter=1e5)

Solves min_X F(X) s.t. X'X = I

Needs the functions F and its gradient dF as well as an initialization satisfying
the orthogonal constraint X0'X0 = I.
"""
function curvilinear_bb(F, dF, X0; τ=1e-2, ρ1=1e-4, ρ2=sqrt(ρ1), δ=0.5, η=0.5, ϵ=1e-3, maxiter=1e3)
    Y(t, A, X) = (I + t/2 * A) \ (I - t/2 * A) * X # Y(τ)
    dY(t, A, X) = -(I + t/2 * A) \ A * (X + Y(t, A, X)) / 2 # Y'(τ)
    dF0(A) = 0.5*norm(A)^2 # F'(Y(τ)) at τ=0 is always this function of A = grad * X' - X * grad'
    dFt(t, A, X) = sum(dF(Y(t, A, X)) .* dY(t, A, X)) # F'(Y(τ))

    X0'X0 ≈ I || ArgumentError("Initial X0 should be orthogonal.") |> throw

    X = X0
    i = 1

    while opnorm(dF(X)) > ϵ && i <= maxiter
        @show i, F(X)

        A = dF(X) * X' - X * (dF(X))'

        # Stepsize conditions 26a, 26b
        while (F(Y(τ, A, X)) >= F(X) + ρ1*τ*dF0(A))# || (dFt(τ, A, X) <= ρ2*dF0(A))
            τ *= δ
            @show τ
            if τ ≈ eps() || τ <= eps() # break if τ is too small
                break
            end
        end

        # Update step
        X = Y(τ, A, X)

        τ /= δ # try to use a larger step next time
        i += 1
    end

    return X # optimal argument value of F(X) s.t. X'X = I
end

U,_,_ = svd(randn(6,6))
X0, _, _ = svd(randn(6,6))

F(X) = 0.5*norm(X - U)^2 #sum((X*X') .* H) # Objective
dF(X) = X-U #H*X # Gradient
curvilinear_bb(F, dF, X0)

X0, _, _ = svd(randn(M,1))
F(X) = -sum(X .* (H*X))
dF(X) = -2*H*X #H*X # Gradient
X1 = curvilinear_bb(F, dF, X0)

H1 = H + F(X1)*X1*X1'
F(X) = -sum(X .* (H1*X))
dF(X) = -2*H1*X #H*X # Gradient
X2 = curvilinear_bb(F, dF, X0)
X = hcat(X1, X2)
X*X'


X0, _, _ = svd(randn(M,5))
X_true, _, _ = svd(randn(M,5))
Z = X_true*X_true'
F(X) = 0.5*norm(X*X' - Z)^2
dF(X) = 2*(X*X' - Z)*X #H*X # Gradient
X = curvilinear_bb(F, dF, X0)

tr(H)
v = eigvals(H)
