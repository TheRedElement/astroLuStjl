"""
astroLuStjl package
"""

module astroLuStjl

#%%imports
using Dates

#metadata
const __modulename__ = "astroLuStjl" 
const __version__ = "1.0.0"
const __author__ = "Lukas Steinwender"
const __author_email__ = ""
const __maintainer__ = "Lukas Steinwender"
const __maintainer_email__ = ""
const __url__ = "https://github.com/TheRedElement/astroLuStjl"
const __credits__ = ""
const __last_changed__ = string(Dates.today())

#add submodules (make visible to parent module)
include("Datascience/Datascience.jl")
include("InOutput/InOutput.jl")
# include("Monitoring/Monitoring.jl")
include("Postprocessing/Postprocessing.jl")
include("Preprocessing/Preprocessing.jl")   #TODO: fails due to method override
include("Styles/Styles.jl")
include("Visualization/Visualization.jl")

#load submodules (make visible to parent module)
using .Datascience
using .InOutput
# using .Monitoring
using .Postprocessing
using .Preprocessing
using .Styles
using .Visualization

#reexport submodules (make accesible to user)
export Datascience
export InOutput
export Monitoring
export Postprocessing
export Preprocessing
export Styles
export Visualization

end #module