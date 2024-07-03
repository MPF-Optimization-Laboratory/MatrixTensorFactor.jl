# MatrixTensorFactor.jl

Overhauled code comming soon! See here for prerelease: https://github.com/MPF-Optimization-Laboratory/MatrixTensorFactor.jl/tree/general-block-decomposition 

New features in testing now:
- More tensor decomposition models like Tucker, Tucker-N, CP and custom factorizations
- More constraints and along any dimentions (ex. columns sum to 1, second order slices are L2 normalized)
- Cyclically or randomly update blocks
- selectively use momentum gradient steps on some blocks
- More convergence criteria (ex. objective value, stationary)
- Record any number of stats every iteration
- Pass in custom initilization 

Planned features:
- More loss functions to optimize (ex. L1 norm, custom objective with auto diff)

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
