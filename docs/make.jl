using Documenter

#push!(LOAD_PATH,"../src/")
#using Pkg
#Pkg.add(url="https://github.com/MPF-Optimization-Laboratory/MatrixTensorFactor.jl.git")
using MatrixTensorFactor
#using SedimentAnalysis
#using SedimentAnalysis.MTF
#using SedimentAnalysis.SedimentTools
#using NamedArrays
using Plots

DocMeta.setdocmeta!(
    SedimentAnalysis,
    :DocTestSetup,
    :(using MatrixTensorFactor;);
    recursive=true
)

makedocs(
    sitename="Matrix Tensor Factorization",
    modules = [MatrixTensorFactor,], #MTF
)

deploydocs(
    repo = "github.com/MPF-Optimization-Laboratory/MatrixTensorFactor.jl.git",
)
