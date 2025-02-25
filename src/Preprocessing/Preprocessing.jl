"""
    - submodule for preprocessing
"""

module Preprocessing

    #include submodules
    include(joinpath(@__DIR__, "./src/ApertureShapes.jl"))
    include(joinpath(@__DIR__, "./src/Bezier3Interp.jl"))
    include(joinpath(@__DIR__, "./src/DataBinning.jl"))     #TODO: issues due to function extension
    include(joinpath(@__DIR__, "./src/OutlierRemoval.jl"))
    include(joinpath(@__DIR__, "./src/Subsampling.jl"))
    include(joinpath(@__DIR__, "./src/Scaling.jl"))
    
    #load submodules (relative reference)
    using .ApertureShapes
    using .Bezier3Interp
    using .DataBinning
    using .OutlierRemoval
    using .Subsampling
    using .Scaling

    #reexport submodules (make visible to parent module)
    export ApertureShapes
    export Bezier3Interp
    export DataBinning
    export OutlierRemoval
    export Subsampling
    export Scaling
    
    # println(which(DataBinning.generate_bins_int, (Vector,)))
    # println(methods(DataBinning.generate_bins_int))

end #module