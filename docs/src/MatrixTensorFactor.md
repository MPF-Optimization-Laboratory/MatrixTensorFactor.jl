# Exported Terms

## Types
```@docs
Abstract3Tensor
```

## Nonnegative Matrix-Tensor Factorization
```@docs
nnmtf
```

### Implimented Factorization Options
```@docs
IMPLIMENTED_OPTIONS
IMPLIMENTED_NORMALIZATIONS IMPLIMENTED_PROJECTIONS IMPLIMENTED_CRITERIA IMPLIMENTED_STEPSIZES
MIN_STEP
MAX_STEP
```

## Kernel Density Estimation
### 1D
```@docs
default_bandwidth
make_densities standardize_KDEs filter_inner_percentile
```

## 2D
```@docs
repeatcoord
kde2d
coordzip
```

## Approximations
```@docs
d_dx
d2_dx2
curvature
standard_curvature
```

## Other Functions

```@docs
*(::AbstractMatrix, ::Abstract3Tensor)
combined_norm
dist_to_Ncone
plot_factors
rel_error
mean_rel_error
residual
```

## Index

```@index
```
