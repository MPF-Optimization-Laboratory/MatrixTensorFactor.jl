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
    # format = Documenter.HTML(; mathengine=
    #     Documenter.KaTeX(
    #         Dict(
    #             :macros => Dict(
    #                 "\\RR" => "\\mathbb{R}",
    #                 raw"\Xi" => raw"X_{i}",
    #                 raw"\Ru" => raw"R_{\mathrm{univ.}}",
    #                 raw"\Pstd" => raw"P_{\mathrm{std}}",
    #                 raw"\Tstd" => raw"T_{\mathrm{std}}",
    #             ),
    #         )
    #     )
    # )

# \newcommand{\tensor}[1]{\ensuremath{{\boldsymbol{\mathscr{#1}}}}} % tensor
# \newcommand{\mtx}[1]{\ensuremath{\boldsymbol{#1}}} % matrix
)

deploydocs(
    repo = "github.com/MPF-Optimization-Laboratory/MatrixTensorFactor.jl.git",
)
