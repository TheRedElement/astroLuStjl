"""
    - module implementing structs and methods for feature scaling
    
    Structs
    -------
        - `MinMaxScaler`
    
    Functions
    ---------
        - `fit()`
        - `transform()`

    Extended Functions
    ------------------
        - `Plots.plot()`
        - `Plots.plot!()`
    
    Dependencies
    ------------
        - `LaTeXStrings`
        - `NaNStatistics`
        - `Plots`

    Comments
    --------

    Examples
    --------
        - see [Scaling_demo.jl](../../demos/Preprocessing/Scaling_demo.jl)
"""

module Scaling

#%%imports
using LaTeXStrings
using NaNStatistics
using Plots

#import for extending
import Plots: plot, plot!

#intradependencies

#%%exports
export MinMaxScaler
export fit
export transform
export plot, plot!

#%%definitions
"""
    - struct defining the MinMaxScaler

    Fields
    ------
        - `targmin`
            - `Real`, optional
            - minimum of the target range
            - the default is `0.0`
        - `targmax`
            - `Real`, optional
            - maximum of the target range
            - the default is `1.0`
        - `datamin`
            - `Real`, optional
            - minimum of the reference series (series the scaler is fitted to)
            - will be used for transforming new data
            - the default is `nothing`
                - value of the unfitted scaler
        - `datamax`
            - `Real`, optional
            - maximum of the reference series (series the scaler is fitted to)
            - will be used for transforming new data
            - the default is `nothing`
                - value of the unfitted scaler
        - `state`
            - `Symbol`, optional
            - can be one of
            - `:init`
                - the model has been initialized
            - `:fitted`
                - the model has been fitted
            - the default is `:init`

    Methods
    -------
        - `fit()`
        - `transform()`
        - `Plots.plot()`
        - `Plots.plot!()`

    Comments
    --------
"""
struct MinMaxScaler{T <: Real}
    targmin::T
    targmax::T
    datamin::Union{Real,Nothing}
    datamax::Union{Real,Nothing}
    state::Symbol

    #inner constructor (to allow default values)
    function MinMaxScaler(
        targmin::T=0.0,
        targmax::T=1.0
        ;
        datamin::Union{Real,Nothing}=nothing,
        datamax::Union{Real,Nothing}=nothing,
        state::Symbol=:init,
        ) where {T <: Real}

        @assert state in [:init,:fitted] ArgumentError("`state` has to be one of `:init`, `:fitted` but is `$state`!")

        new{T}(targmin, targmax, datamin, datamax, state)    
    end
end

"""
    - function to fit `MinMaxScaler` to data

    Parameters
    ----------
        - `mms`
            - `MinMaxScaler`
            - struct containing hyperparameters of the model
        - `x`
            - `Vector{T <: Real}`
            - data/feature to be used as reference for scaling
                - often the data to be scaled is used
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`
    
    Raises
    ------

    Returns
    -------
        - `mms`
            - `MinMaxScaler`
            - fitted version of `mms`
                - `state` set to `:fitted`

    Comments
    --------
"""
function fit(
    mms::MinMaxScaler,
    x::Vector{T};
    verbose::Int=0,
    )::MinMaxScaler where {T <: Real}

    #fit
    datamin = nanminimum(x)
    datamax = nanmaximum(x)

    return MinMaxScaler(mms.targmin, mms.targmax; datamin=datamin, datamax=datamax, state=:fitted)
end
"""
    - function to transform data based on parameters stored in `MinMaxScaler`

    Parameters
    ----------
        - `mms`
            - `MinMaxScaler`
            - fitted version of `mms`
                - `state` set to `:fitted`
        - `x`
            - `Vector{Real}`
            - data/feature to be scaled
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`

    Raises
    ------
        - `AssertionError`
            - if the model has not been fitted yet

    Returns
    -------
        - `x_scaled`
            - `Vector`
            - scaled version of input `x`
            - has same size as `x`

    Comments
    --------
"""
function transform(
    mms::MinMaxScaler,
    x::Vector{T};
    verbose::Int=0,
    )::Vector where{T <: Real}
    
    @assert mms.state == :fitted "`mms` has not been fitted yet. make sure to call `fit(mms,...)` before transforming"

    #transform
    x_std = (x .- mms.datamin) ./ (mms.datamax - mms.datamin)
    x_scaled = x_std .* (mms.targmax-mms.targmin) .+ mms.targmin

    #check if only one one unique value in series (in that case return the halfway point of the target-range)
    x_scaled = mms.datamax == mms.datamin ? zeros(size(x)).+0.5*(mms.xmax-mms.xmin) : x_scaled

    return x_scaled
end

"""
    - extensions to `Plots.plot!()` and `Plots.plot()`
    - plots result obtained using `MinMaxScaler`

    Parameters
    ----------
        - `plt`
            - `Plots.Plot`
            - panel to plot into
        - `mms`
            - `MinMaxScaler`
            - fitted instance of `MinMaxScaler`
        - `x`
            - `Vector`, optional
            - input data to be shown in a transformed manner

    Raises
    ------

    Returns
    -------
    - `plt`
        - `Plots.Plot`
        - created panel

    Comments
    --------
"""
function Plots.plot!(plt::Plots.Plot,
    mms::MinMaxScaler,
    x::Vector,
    )
        
    #get scaled result
    x_scaled = transform(mms, x)

    #plotting
    scatter!(plt, x; label="Input")
    scatter!(plt,
        x_scaled;
        label="Scaled Input"
    )

end
function Plots.plot(
    mms::MinMaxScaler,
    x::Vector,
    )::Plots.Plot
    plt = plot(;xlabel=L"$x$", ylabel=L"$y$", title="MinMaxScaler")
    plot!(plt, mms, x,)
    return plt
end

end #module