"""
    - submodule defining various common plot types and layouts
"""

module Visualization

#include submodules (relative reference)
include("src/PlotTypes.jl")

#load submodules
using .PlotTypes

#reexport submodules (make visible to parent module)
export PlotTypes

end #module
