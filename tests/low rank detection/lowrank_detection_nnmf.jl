using LinearAlgebra

using Pkg
Pkg.add("Plots")
using Plots
I, J = 60, 20
K=1
M = min(I,J)
R = 10

A = (rand(I, R))
A = A ./ sum.(eachrow(A))
B = (rand(R, J))
B = B ./ sum.(eachrow(B))

#Y = A*B + 0.0003*randn(I,J)
Y = [1 1 0 0;
     1 0 1 0;
     0 1 0 1;
     0 0 1 1
] / 2 #A*B + 0.0003*randn(I,J)
Z = abs.(rand(3,3))
z = -Z[1,:]+Z[2,:]+Z[3,:]
Z = vcat(Z,z')
Y = hcat(Y,Z)

y1 = Y[1,:]'
y2 = Y[2,:]'
y3 = Y[3,:]'
y4 = Y[4,:]'

Y = vcat(Y,y1+y2, y2+y4, y1+2*y2+3*y3)

Y = Y ./ sum.(eachrow(Y))


I,J=size(Y)

ϵ = 0.0001
Z = ϵ*randn(I,J)
Y += Z#add noise

Y = abs.(Y)
Y = Y ./ sum.(eachrow(Y))

Y = [1 1 0 0 1;
     1 0 1 0 1;
     0 1 0 1 1;
     0 0 1 1 1;
     1 1 1 1 2
]


R=4

B = vcat(hcat(zeros(I,I), Y), hcat(Y', zeros(J,J)))

sum.(eachrow(Y))
#heatmap(B)
#heatmap(A)
#heatmap(Y)
σ = svdvals(Y)
plot(σ[1:end-1] ./ σ[2:end])
Z = zero(Y)
Z[6,6] = 0.0005
σ = svdvals(Y+Z)
eigvals(B)

Y = reshape(Y, (size(Y)...,1))

using MatrixTensorFactor
M=min(I,J)
final_loss = zeros(M)

for r in 1:M
    @show r
    C, F, rel_errors, norm_grad, dist_Ncone = nnmtf(Y,r;
        tol=1e-5 / sqrt(r*(I+J*K)),
        projection=:nnscale,
        normalize=:slices,
        stepsize=:lipshitz,
        momentum=true,
        delta=0.8,
        criterion=:ncone,
        online_rank_estimation=true,
        maxiter = 500)
    final_loss[r] = norm(Y-C*F)
end

C, F, rel_errors, norm_grad, dist_Ncone = nnmtf(Y,4;
        tol=1e-5 / sqrt(5*(I+J*K)),
        projection=:nnscale,
        normalize=:slices,
        stepsize=:lipshitz,
        momentum=true,
        delta=0.8,
        criterion=:ncone,
        online_rank_estimation=true,
        maxiter = 500)

#U, σ, V = svd(Y)

# best rank 1 approximation
# solves || Y - X ||_F^2 s.t. rank(X)=1
#Σ = Diagonal(zero(σ))
#Σ[1,1] = σ[1]
#X = U*Σ*V

#0.5 * norm(Y - X)^2

#U, σ, V = svd(Y)

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

function circumscribed_standard_curvature(y)
    n = length(y)
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

partial_sum = [sum(σ[i:M] .^ 2) .^0.5 for i in 2:M]
push!(partial_sum,0)
#partial_sum = final_loss
# errors == partial_sum is true since we are off by the singular values

plot(partial_sum ./ maximum(partial_sum); color=:blue)
plot!(sqrt.(partial_sum) ./ maximum(sqrt.(partial_sum)); color=:orange) |> display
plot!(final_loss ./ maximum(final_loss); color=:green)
normalized_sum = (partial_sum) ./ maximum((partial_sum))

partial_sum1 = [sum(σ[i:M]) for i in 2:M]
diff(sqrt.(partial_sum))
σ


plot(curvature(partial_sum; order=4),title="Curvature") |> display
plot(standard_curvature(partial_sum; order=4),title="Standard Curvature") |> display
plot(curvature(normalized_sum; order=4),title="Normalized Root Curvature") |> display
plot(standard_curvature(normalized_sum; order=4),title="Standard Root Curvature") |> display
plot(circumscribed_standard_curvature(partial_sum),title="Standard Circumscribed Curvature") |> display
plot(circumscribed_standard_curvature(normalized_sum),title="Standard Circumscribed Root Curvature") |> display

plot(curvature(final_loss; order=4),title="Curvature") |> display
plot(standard_curvature(final_loss; order=4),title="Curvature") |> display


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
