"""
    - submodule defining various styles and formats
"""

module Styles

#include submodules (relative reference)
include("src/FormattingUtils.jl")
include("src/PlotStyleLuSt.jl")

#load submodules
using .FormattingUtils
using .PlotStyleLuSt

#reexport submodules (make visible to parent module)
export FormattingUtils
export PlotStyleLuSt

end #module
