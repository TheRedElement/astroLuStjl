"""
    - submodule for postprocessing
"""

module Postprocessing

    #include submodules
    include(joinpath(@__DIR__, "./src/Steganography.jl"))
    
    #load submodules (relative reference)
    using .Steganography

    #reexport submodules (make visible to parent module)
    export Steganography
end #module