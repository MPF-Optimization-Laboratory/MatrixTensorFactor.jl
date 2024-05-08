# Exported Terms
```@docs
MatrixTensorFactor
```

## Types
```@docs
Abstract3Tensor
```

## Nonnegative Matrix-Tensor Factorization
```@docs
nnmtf
nnmtf_proxgrad_online
```

### Implimented Factorization Options
```@docs
IMPLIMENTED_OPTIONS
IMPLIMENTED_NORMALIZATIONS
IMPLIMENTED_PROJECTIONS
IMPLIMENTED_CRITERIA
IMPLIMENTED_STEPSIZES
```

## Constants
```@docs
MAX_STEP
MIN_STEP
```

## Kernel Density Estimation
### Constants
```@docs
DEFAULT_N_SAMPLES
DEFAULT_ALPHA
```

### 1D
```@docs
default_bandwidth
make_densities
make_densities2d
standardize_KDEs
standardize_2d_KDEs
filter_inner_percentile
filter_2d_inner_percentile
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
rel_error
mean_rel_error
residual
```

## Index

```@index
```
