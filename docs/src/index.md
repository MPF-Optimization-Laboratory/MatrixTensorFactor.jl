# Matrix Tensor Factorization

```@contents
Depth = 3
```
## How setup the environment

### Recomended Method
1. Run `julia`
2. Add the package with `pkg> add https://github.com/MPF-Optimization-Laboratory/MatrixTensorFactor.jl.git` (use `julia> ]` to get to the package manager)
3. Import with `using MatrixTensorFactor`

**OR**
### In Browser
1. Go to https://github.com/MPF-Optimization-Laboratory/MatrixTensorFactor.jl
2. Click "<> Code" and press "+" to "Create a codespace on main". It make take a few moments to set up.
3. Open the command palett with `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac)
4. Enter `>Julia: Start REPL`
5. In the REPL, resolve any dependency issues with `pkg> resolve` and `pkg> instantiate` (use `julia> ]` to get to the package manager). It may take a few minutes to download dependencies.

Run one of the example files by opening the file and pressing the triangular "run" button, or `>Julia: Execute active File in REPL`.

**OR**
### On your own device
1. Clone the repo at https://github.com/MPF-Optimization-Laboratory/MatrixTensorFactor.jl
2. Navigate to the root of the repository in a terminal and run `julia`
3. Activate the project with `pkg> activate .` (use `julia> ]` to get to the package manager)
4. resolve any dependency issues with `pkg> resolve`

### Importing the package
Type `julia> using MatrixTensorFactor`

## Examples
`smalldata`: decomposes a subset of genomic data to identify gene profiles for learned cell types
`syntheticdata1d.jl`: generate multiple mixtures of 3, 1d probability distributions
`syntheticdata2d.jl`: generate multiple mixtures of 3, 2d probability distributions

## MatrixTensorFactor
Defines the main factorization function [`nnmtf`](@ref) and related mathematical functions. See the full list of [Exported Terms](@ref).

## Index

```@index
```
