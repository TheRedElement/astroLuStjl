#TODO: documentation
"""
    - module implementing ...

    
    Structs
    -------
    
    Functions
    ---------


    Extended Functions
    ------------------

    
    Dependencies
    ------------
        - `NaNStatistics`


    Examples
    --------
"""

module Scaling

#%%imports
using NaNStatistics

#import for extending
import Plots: plot, plot!

#intradependencies

#%%definitions
"""
"""
#TODO: transform to struct with `fit()` and `transform()`
function minmaxscaler(
    x::AbstractArray, xmin::T=0.0, xmax::T=1.0;
    )::AbstractArray where {T <: Real}

    #fit
    datamax = nanmaximum(x)
    datamin = nanminimum(x)

    #transform
    x_std = (x .- datamin) ./ (datamax - datamin)
    x_out = x_std .* (xmax-xmin) .+ xmin

    #check if only one one unique value in series (in that case return the halfway point of the target-range)
    x_out = datamax == datamin ? zeros(size(x)).+0.5*(xmax-xmin) : x_out

    return x_out
end


end #module