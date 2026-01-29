using CaProcessing
using Documenter

DocMeta.setdocmeta!(CaProcessing, :DocTestSetup, :(using CaProcessing); recursive=true)

makedocs(;
    modules=[CaProcessing],
    authors="Galen Lynch <galen@galenlynch.com>",
    sitename="CaProcessing.jl",
    format=Documenter.HTML(;
        canonical="https://galenlynch.github.io/CaProcessing.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/galenlynch/CaProcessing.jl",
    devbranch="main",
)
