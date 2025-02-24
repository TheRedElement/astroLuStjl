"""
    - submodule for preprocessing
"""

module Preprocessing

    #include submodules
    include("src/ApertureShapes.jl")
    include("src/Bezier3Interp.jl")
    include("src/DataBinning.jl")
    include("src/OutlierRemoval.jl")
    include("src/Subsampling.jl")
    include("src/Scaling.jl")
    include("src/Steganography.jl")
    
    #load submodules (relative reference)
    using .ApertureShapes
    using .Bezier3Interp
    using .DataBinning
    using .OutlierRemoval
    using .Preprocessing
    using .Scaling
    using .Steganography

    #reexport submodules (make visible to parent module)
    export ApertureShapes
    export Bezier3Interp
    export DataBinning
    export OutlierRemoval
    export Preprocessing
    export Scaling
    export Steganography
end #module