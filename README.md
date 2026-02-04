# MatrixTensorFactor.jl

This package has been deprecated. Please use the newer and more robust BlockTensorFactorization.jl package: https://github.com/MPF-Optimization-Laboratory/BlockTensorFactorization.jl.

## About

See documentation here: https://mpf-optimization-laboratory.github.io/MatrixTensorFactor.jl/dev/

Factorize a 3rd order tensor Y into a matrix A and 3rd order tensor B: Y=AB. Componentwise, this is:

Y[i,j,k] = sum_{r=1}^R ( A[i,r] * B[r,j,k] )

# Reference
If you find this package at all helpful, please cite the associated paper which is avalible for preprint here:
https://friedlander.io/publications/2024-sediment-source-analysis/

```
@misc{graham_tracing_2024,
	title = {Tracing {Sedimentary} {Origins} in {Multivariate} {Geochronology} via {Constrained} {Tensor} {Factorization}},
	url = {https://friedlander.io/publications/2024-sediment-source-analysis/},
	urldate = {2024-06-28},
	author = {Graham, Naomi and Richardson, Nicholas and Friedlander, Michael P. and Saylor, Joel},
	month = may,
	year = {2024},
  note = {Preprint},
  journal = {Preprint},
}
```

