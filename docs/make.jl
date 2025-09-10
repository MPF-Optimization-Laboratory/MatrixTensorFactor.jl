using Documenter

#push!(LOAD_PATH,"../src/")

using BlockTensorFactorization

DocMeta.setdocmeta!(
    BlockTensorFactorization,
    :DocTestSetup,
    :(using BlockTensorFactorization;);
    recursive=true
)

makedocs(
    sitename="Block Tensor Decomposition",
    modules = [BlockTensorFactorization,],
    checkdocs=:exports,
)

deploydocs(
    repo = "github.com/MPF-Optimization-Laboratory/BlockTensorFactorization.jl.git",
)
