"""
    - submodule for system and execution monitoring
"""

module Monitoring

    #include submodules
    include("src/Timing.jl")
    
    #load submodules (relative reference)
    using .Timing

    #reexport submodules (make visible to parent module)
    export Timing
end #module