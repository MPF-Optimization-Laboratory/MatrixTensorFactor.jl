using LinearAlgebra
using Plots

I, J = 100, 80
M = min(I,J)
R = 20

A = (rand(I, R))
A = A ./ sum.(eachrow(A))
B = (rand(R, J))
B = B ./ sum.(eachrow(B))

Y = A*B + 0.0003*randn(I,J)

sum.(eachrow(Y))
heatmap(B)
heatmap(A)
heatmap(Y)

#U, σ, V = svd(Y)

# best rank 1 approximation
# solves || Y - X ||_F^2 s.t. rank(X)=1
#Σ = Diagonal(zero(σ))
#Σ[1,1] = σ[1]
#X = U*Σ*V

#0.5 * norm(Y - X)^2

U, σ, V = svd(Y)

#=
function low_rank_approx(rank=1)

    # best rank r approximation
    # solves || Y - X ||_F^2 s.t. rank(X)=1
    Σ = Diagonal(zero(σ))
    for i in 1:rank
        Σ[i,i] = σ[i]
    end
    X = U*Σ*V'
    error = 0.5 * norm(Y - X)^2

    return X, error
end

errors = zeros(min(I,J))
for r in eachindex(errors)
    _, error = low_rank_approx(r)
    errors[r] = error
end

using Plots
plot(errors)
=#

partial_sum = [0.5*sum(σ[i:M] .^ 2) for i in 2:M]
push!(partial_sum,0)

# errors == partial_sum is true since we are off by the singular values

plot(partial_sum ./ maximum(partial_sum); color=:blue)
plot!(sqrt.(partial_sum) ./ maximum(sqrt.(partial_sum)); color=:orange) |> display
normalized_sum = sqrt.(partial_sum) ./ maximum(sqrt.(partial_sum))

partial_sum1 = [sum(σ[i:M]) for i in 2:M]
diff(sqrt.(partial_sum))
σ


using MatrixTensorFactor

plot(curvature(partial_sum; order=4),title="Curvature") |> display
plot(standard_curvature(partial_sum; order=4),title="Standard Curvature") |> display
plot(curvature(normalized_sum; order=4),title="Normalized Root Curvature") |> display
plot(standard_curvature(normalized_sum; order=4),title="Standard Root Curvature") |> display
plot(circumscribed_standard_curvature(partial_sum),title="Standard Circumscribed Curvature") |> display
plot(circumscribed_standard_curvature(normalized_sum),title="Standard Circumscribed Root Curvature") |> display

best_c(u,v) = u'v / norm(u)^2 # finds c = argmin_c 0.5 || v - cu ||_2^2

dist(u,v) = 0.5 * norm(v - best_c(u,v)*u)^2

#=
for (j, u) in enumerate(eachrow(B))
    for (i, v) in enumerate(eachrow(B))
        if v == u || j < i
            continue
        end
        scatter(u,v; title="$i vs $j with dist=$(dist(u,v))")
        a = min(minimum(u),minimum(v))
        b = max(maximum(u),maximum(v))
        plot!([a,b], [a,b]) |> display
    end
end
=#
#=
u = B[2,:]
v = B[4,:]

scatter(v, u)
a = min(minimum(u),minimum(v))
b = max(maximum(u),maximum(v))
plot!([a,b], [a,b]) |> display
=#

function circumscribed_standard_curvature(y)
    n = length(v)
    ymax = maximum(y)
    y = y / ymax
    k = zero(y)
    a, b, c = 0, 1/n, 2/n
    for i in eachindex(k)[2:end-1]
        k[i] = 1 / circumscribed_radius((a,y[i-1]),(b,y[i]),(c,y[i+1]))
    end
    k[1] = k[2]
    k[end] = k[end-1]
    return k
end

function circumscribed_radius((a,f),(b,g),(c,h))
    d = 2*(a*(g-h)+b*(h-f)+c*(f-g))
    p = ((a^2+f^2)*(g-h)+(b^2+g^2)*(h-f)+(c^2+h^2)*(f-g)) / d
    q = ((a^2+f^2)*(b-c)+(b^2+g^2)*(c-a)+(c^2+h^2)*(a-b)) / d
    r = sqrt((a-p)^2+(f-q)^2)
    return r
end
