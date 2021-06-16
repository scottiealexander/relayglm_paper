using Pkg

"""
install_dependencies(local_env::Bool=true)
    local_env::Bool - set to false to install dependencies to the home project
"""
function install_dependencies(local_env::Bool=true)
    if local_env
        if !isfile(joinpath(@__DIR__, "Project.toml"))
            open(joinpath(@__DIR__, "setpath.jl"), "a") do io
                write(io, "using Pkg\nPkg.activate(@__DIR__)\n")
            end
        end
        Pkg.activate(@__DIR__)
    end

    # registered dependencies
    Pkg.add([
        "LinearAlgebra",
        "Statistics",
        "Random",
        "Printf",
        "Dates",
        "JSON",
        "PyPlot",
        "PyCall",
        "Bootstrap",
        "ImageFiltering",
        "StatsBase",
        "Optim",
        "Distributions",
        "Colors",
        "ColorTypes",
        "KernelDensity"
    ])

    # unregistered dependencies
    Pkg.add(PackageSpec(url="https://github.com/scottiealexander/SpkCore.jl.git"))
    Pkg.add(PackageSpec(url="https://github.com/scottiealexander/RelayGLM.jl.git"))
    Pkg.add(PackageSpec(url="https://github.com/scottiealexander/PairsDB.jl.git"))
end
