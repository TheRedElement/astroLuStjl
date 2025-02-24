"""
    - submodule for standard input and output routines
"""


module InOutput

    #include submodules
    include("src/FileInOutput.jl")
    include("src/StringParsing.jl")
    
    #load submodules (relative reference)
    using .FileInOutput
    using .StringParsing

    #reexport submodules (make visible to parent module)
    export FileInOutput
    export StringParsing

end #module