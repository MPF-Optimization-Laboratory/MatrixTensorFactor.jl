using Documenter

#push!(LOAD_PATH,"../src/")

using MatrixTensorFactor

DocMeta.setdocmeta!(
    MatrixTensorFactor,
    :DocTestSetup,
    :(using MatrixTensorFactor;);
    recursive=true
)

makedocs(
    sitename="Matrix Tensor Factorization",
    modules = [MatrixTensorFactor,], #MTF
    checkdocs=:exports,
)

deploydocs(
    repo = "github.com/MPF-Optimization-Laboratory/MatrixTensorFactor.jl.git",
)
