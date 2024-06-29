using Documenter

#push!(LOAD_PATH,"../src/")

using BlockTensorDecomposition

DocMeta.setdocmeta!(
    BlockTensorDecomposition,
    :DocTestSetup,
    :(using BlockTensorDecomposition;);
    recursive=true
)

makedocs(
    sitename="Block Tensor Decomposition",
    modules = [BlockTensorDecomposition,],
    checkdocs=:exports,
)

deploydocs(
    repo = "github.com/MPF-Optimization-Laboratory/BlockTensorDecomposition.jl.git",
)
