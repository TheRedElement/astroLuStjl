"""
    - submodule for standard input and output routines
"""


module InOutput

    #include submodules
    include(joinpath(@__DIR__, "./src/FileInOutput.jl"))
    include(joinpath(@__DIR__, "./src/StringParsing.jl"))
    
    #load submodules (relative reference)
    using .FileInOutput
    using .StringParsing

    #reexport submodules (make visible to parent module)
    export FileInOutput
    export StringParsing

end #module