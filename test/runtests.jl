
#%%imports
using Pkg
using Plots

Pkg.activate(joinpath(@__DIR__, "."))                       #activate env (init new one if nonexistent)
Pkg.rm("astroLuStjl")
Pkg.develop(path=joinpath(@__DIR__, "../../astroLuStjl/"))  #snapshot of local directory
# Pkg.add(path=joinpath(@__DIR__, "../../astroLuStjl/"))  #snapshot of git code


using astroLuStjl

#%%main
println("test complete")

# println(Styles.mono_colors)
# theme(:tre_dark)
# theme(:tre_light)

# p = scatter(randn(80), randn(80))
# display(p)
