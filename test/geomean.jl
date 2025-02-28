using BenchmarkTools
using Random

f1(x) = prod(x)^(1/length(x))
f2(x) = exp(mean(log.(x)))

absrandn(x) = abs.(randn(x))
v = absrandn(10)

f1(v) ≈ f2(v)

@benchmark f1(x) setup=(x = absrandn(10000))
@benchmark f1(x) setup=(x = 100000*absrandn(10000))
@benchmark f2(x) setup=(x = absrandn(10000))
@benchmark f2(x) setup=(x = 100000*absrandn(10000))

function geomean1(v)
    p = prod(v)
    if isinf(p) | iszero(p)
        return exp(mean(log.(v)))
    else
        # Faster, but more sensitive
        return p^(1/length(v))
    end
end

function geomean2(v)
    p = prod(v)
    if iszero(p) | isinf(p)
        return exp(mean(log.(v)))
    else
        # Faster, but more sensitive
        return p^(1/length(v))
    end
end

@benchmark geomean1(x) setup=(x = absrandn(10000))
@benchmark geomean2(x) setup=(x = absrandn(10000))
@benchmark geomean1(x) setup=(x = 10*absrandn(10000))
@benchmark geomean2(x) setup=(x = 10*absrandn(10000))

@benchmark geomean1(x) setup=(x = 10*absrandn(10))
@benchmark f1(x) setup=(x = 10*absrandn(10))
@benchmark f2(x) setup=(x = 10*absrandn(10))
