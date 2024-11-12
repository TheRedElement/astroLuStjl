"""
    - module implementing data binning

    
    Structs
    -------
        - `Binning`
        - `BinningResult`
    
    Functions
    ---------
        - `generate_bins_int()`
        - `generate_bins_pts()`
        - `generate_bins_centered()`
        - `generate_bins_phys()`
        - `fit()`
        - `transform()`

    Extended Functions
    ------------------
        - `Plots.plot()`
        - `Plots.plot!()`
    
    Dependencies
    ------------
        - `Iterators`   
        - `LaTeXStrings`   
        - `NaNStatistics`
        - `Plots`
        - `StatsBase`

    Examples
    --------
        - see [../src_demos/Binning_demo.jl](../src_demos/Binning_demo.jl)
    
"""

#%%imports
using LaTeXStrings
using NaNStatistics
using Plots
using StatsBase

#import for extending
import Plots: plot, plot!

#intradependencies

#%%definitions

#######################################
#bin-generation
"""
    - function to generate a vector of bin boundaries
    - constant bin-widths, dynamic density

    Parameters
    ----------
        - `x`
            - `Vector`
            - x-values to be used as reference for creating the bins
        - `nintervals`
            - `AbstractFloat`, `Int`, optional
            - number of intervals to generate
            - if `AbstractFloat`
                - will be interpreted as fraction of `size(x,1)`
        - `xmin`
            - `Real`, `Nothing`, optional
            - minimum value the generated bins shall have
            - the default is `nothing`
                - will be set to `minimum(x)`
        - `xmax`
            - `Real`, `Nothing`, optional
            - maximum value the generated bins shall have
            - the default is `nothing`
                - will be set to `maximum(x)`
        - `eps`
            - `AbstractFloat`, optional
            - constant to add to `xmax`
            - to ensure that the bin-boundary is strictly greater than the `maximum(x)`
            - the default is `1e-6`
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`

    Raises
    ------

    Returns
    -------
        - `bins`
            - `Vector{AbstractFloat}`
            - generated bin boundaries

    Comments
    --------
"""
function generate_bins_int(
    x::Vector, nintervals::AbstractFloat=0.2;
    xmin::Union{Nothing,Real}=nothing, xmax::Union{Nothing,Real}=nothing,
    eps::AbstractFloat=1e-6,
    verbose::Int=0,
    )::Vector{AbstractFloat}

    @assert (0 < nintervals)&(nintervals < 1) "`nintervals` has to be between 0 and 1!"

    #default values
    xmin = ~isnothing(xmin) ? xmin : nanminimum(x)
    xmax = ~isnothing(xmax) ? xmax : nanmaximum(x)
    
    nintervals = Int(round(nintervals*size(x,1); digits=0)) #infer number of intervals

    #generate bins #convert to `AbstractFloat` for type consistency
    bins = AbstractFloat.(range(xmin, xmax+eps, nintervals+1))  #nintervals+1 since this defines the bin boundaries
    
    if verbose > 0
        p = plot(x, ones(size(x)), seriestype=:scatter, label="Input")
        vline!(p, bins, label="Bins")
        display(p)
    end
    return bins
end
function generate_bins_int(
    x::Vector, nintervals::Int=10;
    xmin::Union{Nothing,Real}=nothing, xmax::Union{Nothing,Real}=nothing,
    eps::AbstractFloat=1e-6,
    verbose::Int=0,
    )::Vector{AbstractFloat}

    #requirements
    @assert (0 < nintervals)&(nintervals < size(x,1)) "`nintervals` has to be between 0 and `size(x,1)`!"

    #default values
    xmin = ~isnothing(xmin) ? xmin : nanminimum(x)
    xmax = ~isnothing(xmax) ? xmax : nanmaximum(x)

    #generate bins #convert to `AbstractFloat` for type consistency
    bins = AbstractFloat.(range(xmin, xmax+eps, nintervals+1))  #nintervals+1 since this defines the bin boundaries
    
    if verbose > 0
        p = plot(x, ones(size(x)), seriestype=:scatter, label="Input")
        vline!(p, bins, label="Bins")
        display(p)
    end
    return bins
end

"""
    - function to generate a vector of bin boundaries
    - constant density, dynamic bin-widths

    Parameters
    ----------
        - `x`
            - `Vector`
            - x-values to be used as reference for creating the bins
        - `npointsperinterval`
            - `AbstractFloat`, `Int`, optional
            - number of points each generated bin shall contain
            - if `AbstractFloat`
                - will be interpreted as fraction of `size(x,1)`
        - `eps`
            - `AbstractFloat`, optional
            - constant to add to `xmax`
            - to ensure that the bin-boundary is strictly greater than the `maximum(x)`
            - the default is `1e-6`        
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`

    Raises
    ------

    Returns
    -------
        - `bins`
            - `Vector{AbstractFloat}`
            - generated bin boundaries

    Comments
    --------
"""
function generate_bins_pts(
    x::Vector, npointsperinterval::Int=10;
    eps::AbstractFloat=1e-6,
    verbose::Int=0,
    )::Vector{AbstractFloat}

    #partition
    bins = Iterators.partition(x,npointsperinterval)
    
    #obtain bin-bounds
    bins = map(x -> AbstractFloat(nanminimum(x)), bins)    #convert to `AbstractFloat` for type consistency
    
    #add offset to ensure bools can be evaluated correctly
    if bins[end] == nanmaximum(x)    #in case the bins happen to align perfectly with the last datapoint (necessary to also consider last datapoint)
        bins[end] += eps
    else                             #general case
        push!(bins, nanmaximum(x)+eps)
    end

    if verbose > 0
        p = plot(x, ones(size(x)), seriestype=:scatter, label="Input")
        vline!(p, bins, label="Bins")
        display(p)
    end
    return bins
end
function generate_bins_pts(
    x::Vector, npointsperinterval::AbstractFloat=0.1;
    eps::AbstractFloat=1e-6,
    verbose::Int=0,
    )::Vector{AbstractFloat}

    npointsperinterval = Int(round(npointsperinterval*size(x,1); digits=0)) #infer number of points per interval

    #partition
    bins = Iterators.partition(x, npointsperinterval)
    
    #obtain bin-bounds
    bins = map(x -> AbstractFloat(nanminimum(x)), bins)    #convert to `AbstractFloat` for type consistency
    
    #add offset to ensure bools can be evaluated correctly
    if bins[end] == nanmaximum(x)    #in case the bins happen to align perfectly with the last datapoint (necessary to also consider last datapoint)
        bins[end] += eps
    else                             #general case
        push!(bins, nanmaximum(x)+eps)
    end

    if verbose > 0
        p = plot(x, ones(size(x)), seriestype=:scatter, label="Input")
        vline!(p, bins, label="Bins")
        display(p)
    end
    return bins
end

"""
    - generate bins of width `dx` given assuming `x` has some physical meaning
    - both `x` and `dx` have to be values of the same physical units

    Parameters
    ----------
        - `x`
            - `Vector`
            - x-values to be used as reference for creating the bins
            - has to have some physical meaning
            - value has to be provided in same units as `dx`
        - `dx`
            - `Real`
            - bin width in physical units of `x`
                - each generated bin (apart from the last one) will cover an interval of size `dx`
            - the last bin will contain all remaining points
        - `eps`
            - `AbstractFloat`, optional
            - constant to add to `xmax`
            - to ensure that the bin-boundary is strictly greater than the `maximum(x)`
            - the default is `1e-6`        
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`

    Raises
    ------

    Returns
    -------
        - `bins`
            - `Vector{AbstractFloat}`
            - generated bin boundaries

    Comments
    --------
"""
function generate_bins_phys(
    x::Vector, dx::Real;
    eps::AbstractFloat=1e-6,
    verbose::Int=0,
    )::Vector{AbstractFloat}

    bins = collect(range(nanminimum(x), nanmaximum(x); step=dx))
    
    #add offset to ensure bools can be evaluated correctly
    if bins[end] == nanmaximum(x)    #in case the bins happen to align perfectly with the last datapoint (necessary to also consider last datapoint)
        bins[end] += eps
    else                             #general case
        push!(bins, nanmaximum(x)+eps)
    end

    return bins

end

"""
    - function to generate a set of bin-edges centered around `centers`

    Parameters
    ----------
        - `centers`
            - `UnitRange`, `Vector`
            - centers to genreate bins around
        - `bin_width`
            - `Real`, optional
            - width each bin shall have as a fraction of the most common difference between two centers (`StatsBase.mode(diff(centers))`)
            - this will just modify the bin-edge position
                - if your data is continuous, also the inbetween spaces will be filled!
            - the default is `1`

    Raises
    ------

    Returns
    -------
        - `bins`
            - `Vector`
            - the generated bin-edges

    Comments
    --------
"""
function generate_bins_centered(
    centers::Union{AbstractRange,Vector};
    bin_width::Real=1.0,
    )::Vector

    #most common step
    step = StatsBase.mode(diff(centers))

    #get bins
    bins = sort([(centers .- (step*bin_width/2))..., (centers .+ (step*bin_width/2))...])

    return bins
end

#######################################
#binning
"""
    - struct defining the data-binning transformer
    
    Fields
    ------
        - `bins`
            - `Vector{Real}`
            - vector containing the boundaries of the bins to use
        - `mean_func_x`
            - `Function`, optional
            - function to be used for the computation of the representative value of each bin in x-direction
            - the default is `NaNStatistics.nanmean`
        - `mean_func_y`
            - `Function`, optional
            - function to be used for the computation of the representative value of each bin in y-direction
            - the default is `NaNStatistics.nanmean`
        - `std_func_x`
            - `Function`, optional
            - function to be used for the computation of the scatter of each bin in x-direction
            - the default is `NaNStatistics.nanstd`
        - `std_func_y`
            - `Function`, optional
            - function to be used for the computation of the scatter of each bin in y-direction
            - the default is `NaNStatistics.nanstd`
    
    Methods
    -------
        - `fit()`
        - `transform()`
        - `Plots.plot()`
        - `Plots.plot!()`

    Related Structs
    ---------------
        - `BinningResult`
    
    Comments
    --------

"""
struct Binning
    bins
    mean_func_x
    mean_func_y
    std_func_x
    std_func_y
    
    #inner constructor (to allow default values)
    function Binning(
        bins::Vector;
        mean_func_x::Function=NaNStatistics.nanmean, mean_func_y::Function=NaNStatistics.nanmean,
        std_func_x::Function=NaNStatistics.nanstd, std_func_y::Function=NaNStatistics.nanstd,
        )

        new(bins, mean_func_x, mean_func_y, std_func_x, std_func_y)

    end
end

"""
    - struct to store result of `Binning`

    Fields
    ------
        - `x_binned`
            - `Vector{Real}`    
            - binned version of the x-values
        - `y_binned`
            - `Vector{Real}`    
            - binned version of the y-values
        - `x_std`
            - `Vector{Real}`
            - scatter contained within the bins of the x-values
        - `y_std`
            - `Vector{Real}`
            - scatter contained within the bins of the y-values
"""
struct BinningResult
    x_binned::Vector{Real}
    y_binned::Vector{Real}
    x_std::Vector{Real}
    y_std::Vector{Real}
end

"""
    - function to fit `Binning` to data
    - i.e. exectues data-binning given the parameters of `Binning`
    
    Parameters
    ----------
        - `bng`
            - `Binning`
            - struct containing the hyperparameters of the model
        - `x`
            - `Vector{Real}`
            - x-values of the dataseries to be binned
        - `y`
            - `Vector{Real}`
            - y-values of the dataseries to be binned
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`
    
    Raises
    ------

    Returns
    -------
        - `br`
            - `BinningResult`
            - struct containing computed quantities

    Comments
    --------
"""
function fit(
    bng::Binning,
    x::Vector{T}, y::Vector{T};
    verbose::Int=0,
    )::BinningResult where {T<:Real}

    nbins = size(bng.bins,1)
    
    #init output
    br = BinningResult(
        Vector(undef,nbins-1), Vector(undef,nbins-1),
        Vector(undef,nbins-1), Vector(undef,nbins-1),
    )

    for (idx, (b1, b2)) in enumerate(zip(bng.bins[1:end-1], bng.bins[2:end]))
        bin_bool = (b1 .<= x) .& (x .<= b2)
        nan_bool = (isfinite.(x)) .& (isfinite.(y))
        br.x_binned[idx]    = bng.mean_func_x(x[bin_bool .& nan_bool])
        br.y_binned[idx]    = bng.mean_func_y(y[bin_bool .& nan_bool])
        br.x_std[idx]       = bng.std_func_x(x[bin_bool .& nan_bool])
        br.y_std[idx]       = bng.std_func_y(y[bin_bool .& nan_bool])
        
    end
    
    return br

end

"""
    - function to transform data based on parameters stored in `BinningResult`
    - i.e. parameters resulting from data-binning
    
    Parameters
    ----------
        - `br`
            - `BinningResult`
            - struct containing parameters inferred by data-binning
        - `x`
            - `Vector{Real}`, `Nothing`
            - x-values of the dataseries to be transformed
            - only for consistency (not needed for evaluation)
            - the default is `nothing`
        - `y`
            - `Vector{Real}`, `Nothing`
            - y-values of the dataseries to be transformed
            - only for consistency (not needed for evaluation)
            - the default is `nothing`
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`
    
    Raises
    ------

    Returns
    -------
        - `x_binned`
            - `AbstractArray`
            - binned version of `x`
        - `y_binned`
            - `AbstractArray`
            - binned version of `y`
        - `x_std`
            - `AbstractArray`
            - std corresponding to `x_binned`
            - i.e., std in x-direction for each bin
        - `x_std`
            - `AbstractArray`
            - std corresponding to `y_binned`
            - i.e., std in y-direction for each bin

    Comments
    --------
"""
function transform(
    br::BinningResult,
    x::Union{Vector,Nothing}=nothing, y::Union{Vector,Nothing}=nothing;
    verbose::Int=0,
    )::NTuple{4, AbstractArray}
    x_binned = br.x_binned
    y_binned = br.y_binned
    x_std = br.x_std
    y_std = br.y_std
    return x_binned, y_binned, x_std, y_std
end

"""
    - extensions to `Plots.plot!()` and `Plots.plot()`

    Parameters
    ----------
        - `plt`
            - `Plots.Plot`
            - panel to plot into    
        - `br`
            - `BinningResult`
            - struct containing the result from `fit(Binning)` that shall be visualized
        - `bins`
            - `Vector`, optional
            - bins corresponding to `br`
            - the default is `nothing`
                - not considered
        - `x`
            - `Vector`, optional
            - original x-values corresponding to `br`
            - the default is `nothing`
                - not considered
        - `y`
            - `Vector`, optional
            - original y-values corresponding to `br`
            - the default is `nothing`
                - not considered

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
    br::BinningResult;
    bins::Union{Vector,Nothing}=nothing, x::Union{Vector,Nothing}=nothing, y::Union{Vector,Nothing}=nothing,
    )

    #current theme defaults
    lw = Plots.default(:linewidth)  #linewidth
    bg = Plots.default(:bg)         #background
    ms = Plots.default(:ms)         #markersize


    #input scatter (only if `x` and ` y` were passed)
    if ~isnothing(x) && ~isnothing(y)
        scatter!(plt, x, y; label="Input")
    end

    #binned curve
    plot!(plt,
        br.x_binned, br.y_binned,
        xerr=br.x_std, yerr=br.y_std,
        ls=:solid, lw=2.5*lw, color=bg, markerstrokecolor=bg, ms=1.5*ms,
        marker=:circle, label=""
    )
    plot!(plt,
        br.x_binned, br.y_binned,
        xerr=br.x_std, yerr=br.y_std,
        marker=:circle, label="Binned"
    )

    if ~isnothing(bins)
        vline!(plt, bins; alpha=0.5, label="Intervals")
    end
end
function Plots.plot(
    br::BinningResult;
    bins::Union{Vector,Nothing}=nothing, x::Union{Vector,Nothing}=nothing, y::Union{Vector,Nothing}=nothing,
    )::Plots.Plot
    plt = plot(; xlabel=L"$x$", ylabel=L"$y$", title="Binning")
    plot!(plt, br; x=x, y=y, bins=bins)
    return plt
end


