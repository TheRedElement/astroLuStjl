"""
    - submodule for data-science algorithms
"""


module Datascience

    #include submodules
    include("src/TimeseriesForecasting.jl")

    #load submodules (relative reference)
    using .TimeseriesForecasting

    #reexport submodules (make visible to parent module)
    export TimeseriesForecasting

end #module