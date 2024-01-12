using Documenter

#push!(LOAD_PATH,"../src/")
#using Pkg
#Pkg.add(url="https://github.com/MPF-Optimization-Laboratory/MatrixTensorFactor.jl.git")
using MatrixTensorFactor
#using SedimentAnalysis
#using SedimentAnalysis.MTF
#using SedimentAnalysis.SedimentTools
#using NamedArrays
#using Plots

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
