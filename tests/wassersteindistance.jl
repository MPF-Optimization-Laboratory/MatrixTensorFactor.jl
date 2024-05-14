using Pkg
Pkg.add("ReverseDiff")
Pkg.add("OptimalTransport")
Pkg.add("Distributions")
Pkg.add("Plots")

using ReverseDiff: gradient, GradientTape, compile, gradient!
using Plots
#using Einsum
#using Statistics
using Distributions
#using SpectralDistances
using OptimalTransport
using LinearAlgebra

#####################
# Reverse Diff test #
#####################
h(x) = 0.5*norm(x)^2
a = [1.0, 2.0, 3.0]

const h_tape = GradientTape(h, similar(a))
const compiled_h_tape = compile(h_tape)

# Allocates space for the output
∇h(input) = gradient!(similar(input), compiled_h_tape, input)
# Pushes results into "results"
∇h!(results, input) = gradient!(results, compiled_h_tape, input)

###################
# Reverse Diff OT #
###################

# Define coordinates
x = range(-5,5,length=20)

# (Descrete) Probability values
a = @. 1/sqrt(2pi)*exp(-0.5*x^2)
b = @. -abs(x)+5.1
a ./= sum(a) # Normalize
b ./= sum(b)

# Wrap using Densities.jl
μ = DiscreteNonParametric(x, a)
ν = DiscreteNonParametric(x, b)

# Define loss functions
function f(a,b)
    μ = DiscreteNonParametric(x, a)
    ν = DiscreteNonParametric(x, b)
    wasserstein(μ, ν) # returns a float
end

const f_tape = GradientTape(f, (a,b))
const compiled_f_tape = compile(f_tape)
#u,v = (similar(a), similar(b))

∇f(a,b) = gradient(compiled_f_tape, (a, b))
grad_a, grad_b = ∇f(a,b)

########
# Plot #
########

plot(μ.support, μ.p, label="μ Probability")
plot!(ν.support, ν.p, label="ν Probability")

plot(x, grad_a, label="∇f_a(a,b)")
plot!(x, grad_b, label="∇f_b(a,b)")

grad_a + grad_b ≈ 0 # The gradients should be opposite by symmetry


"""
function W(C, a, b; β=1e-1, iters=1000)
    ϵ = eps()
    K = exp.(.-C ./ β)
    v = one.(b)
    local u
    for _ in 1:iters
        u = a ./ (K * v .+ ϵ)
        v = b ./ (K' * u .+ ϵ)
    end
    Γ =  u .* K .* v'
    u = -β .* log.(u .+ ϵ)
    u = u/mean(u)
    v = -β .* log.(v .+ ϵ)
    v = v/mean(v)

    Γ, u, v
end

# sinkhorn_log

x = range(-5,5,length=20)
a = @. 1/sqrt(2pi)*exp(-0.5*x^2)
b = @. -abs(x)+5.1

a ./= sum(a)
b ./= sum(b)

p=plot(x,a)
plot!(x,b)

display(p)

d2(x,y) = (x-y)^2

@einsum C[i,j] := d2(x[i], x[j])

μ = DiscreteNonParametric(x, a)
ν = DiscreteNonParametric(x, b)


wasserstein(μ, ν)
#out = OptimalTransport.sinkhorn(a,b,C,0.01)
#Γ, u, v = OptimalTransport.emd2(C,a,b)
#Γ, u, v = OptimalTransport.sinkhorn(ones(20,20),a,b)
input = (a,b)
#result = (zero(a),zero(b))
function f(input)
    μ = DiscreteNonParametric(x, input)
    #Γ, u, v = OptimalTransport.sinkhorn(C,input,b)
    #return sum(C .* Γ)
    return wasserstein(μ, ν)
end

function g(input)
    ν = DiscreteNonParametric(x, input)
    #Γ, u, v = OptimalTransport.sinkhorn(C,input,b)
    #return sum(C .* Γ)
    return wasserstein(μ, ν)
end

ga = gradient(f,a)
gb = gradient(g,b)

_, ga2, gb2 = W(C, a, b)
"""
