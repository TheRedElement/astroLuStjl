
#%%imports
using Pkg
using Plots

#testing with project environment
Pkg.activate(joinpath(@__DIR__, ".."))                       #activate env (init new one if nonexistent)

# #testing install into separate environment 
# Pkg.activate(joinpath(@__DIR__, "."))                       #activate env (init new one if nonexistent)
# Pkg.rm("astroLuStjl")
# Pkg.develop(path=joinpath(@__DIR__, "../../astroLuStjl/"))  #snapshot of local directory
# # Pkg.add(path=joinpath(@__DIR__, "../../astroLuStjl/"))  #snapshot of git code
# Pkg.resolve()
# Pkg.instantiate()

using astroLuStjl

#%%control
demo_files = [
    # "../demos/Datascience/TimeseriesForecasting_demo.jl"
    # "../demos/InOutput/FileInOutput_demo.jl"
    # "../demos/InOutput/StringParsing_demo.jl"
    # # "../demos/Monitoring/Timing_demo.jl"
    # "../demos/Postprocessing/Steganography_demo.jl"
    # "../demos/Preprocessing/ApertureShapes_demo.jl"
    # "../demos/Preprocessing/Bezier3Interp_demo.jl"
    # "../demos/Preprocessing/DataBinning_demo.jl"
    # "../demos/Preprocessing/OutlierRemoval_demo.jl"
    # "../demos/Preprocessing/Scaling_demo.jl"
    # "../demos/Preprocessing/Subsampling_demo.jl"
    # "../demos/Styles/FormattingUtils_demo.jl"
    # "../demos/Styles/PlotStyleLuSt_demo.jl"
    "../demos/Visualization/PlotTypes_demo.jl"
]


#%%main
n_complete = 0
n2test = length(demo_files)
for (idx, fn) in enumerate(demo_files)
    try
        printstyled("    Testing $(idx)/$(n2test) ($fn)...\n"; bold=true, color=:green)
        include(joinpath(@__DIR__, fn)) #run test for current file
        printstyled("    Testing $(idx)/$(n2test) ($fn) completed successfully\n"; bold=true, color=:green)
        global n_complete
        n_complete += 1
    catch e
        printstyled("    Testing $(idx)/$(n2test) ($fn) failed\n"; bold=true, color=:red)
        printstyled("        Error: $e\n"; bold=true, color=:red)
    end
end

if n_complete == n2test
    printstyled(">>>All tests completed successfully\n"; bold=true, color=:green)
else
    printstyled(">>> $(n_complete)/$(n2test) tests completed successfully. "; bold=true, color=:green)
    printstyled("$(n2test-n_complete)/$(n2test) tests failed.\n"; bold=true, color=:red)
end
