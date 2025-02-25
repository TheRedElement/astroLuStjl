
#TODO: `parallel_coords()` add histogram
#TODO: `parallel_coords()` add deal with dates

"""
    - module to define custom plotting functions

    Constants
    ---------

    Functions
    ---------
        - `hatched_histogram()`
        - `hatched_histogram!()`
        - `parallel_coords()`
        - `parallel_coords!()`

    Structs
    -------

    Dependencies
    ------------
        - `CategoricalArrays`
        - `DataFrames`
        - `NaNStatistics`
        - `Plots`
        - `Printf`
        - `PGDPlotsX`

    Comments
    --------
        - the packages is built with `gr()` backend
            - other backends should work, but don't necessarily have to
        - in case you want to use `parallel_coords()` or `parallel_coords!()` with `pgfplotsx`
            - make sure to add the following lines to your script
                - `using PGFPlotsX`
                - `PGFPlotsX.CUSTOM_PREAMBLE = ["\\usepackage{pmboxdraw}", "\\usepackage{graphicx}", "\\newcommand{\\blockfull}{\\scalebox{.4}[1]{\\textblock}}"]`

    Examples
    --------
        - see [PlotTypes.jl](../../demos/visualization/PlotTypes.jl)

"""
module PlotTypes

#%%imports
using CategoricalArrays
using DataFrames
using Interpolations
using NaNStatistics
using Plots
using Printf
using PGFPlotsX

#import for extending
import Plots: plot, plot!

#intradependencies
include(joinpath(@__DIR__, "../../Preprocessing/Preprocessing.jl"))
using .Preprocessing.Scaling
using .Preprocessing.Bezier3Interp


#%%exports
export hatched_histogram
export hatched_histogram!
export parallel_coords
export parallel_coords!

#definitions
#######################################
#helper functions
sigmoid(x, tau=1) = 1/(1 + exp(-x/tau))
function  minmaxscaler(x)
    mms = Scaling.MinMaxScaler()
    mms = Scaling.fit(mms, x)
    x_scaled = Scaling.transform(mms, x)
    return x_scaled
end

#######################################
#hatched histogram
"""
    - function to add custom hatching to a series generated by `Plots.histogram()`
    - allows for more artistic freedom compared to `Plots.histogram(fillstyle=...)`

    Parameters
    ----------
        - `hg`
            - `Plots.Plot`
            - histogram to add the hatching to
            - has to be generated with `Plots.histogram()`
        - `k`
            - `Real`, optional
            - slope of the hatching lines
            - the default is `0.1`
                - works well for normalized histograms (`Plots.histogram(..., normalize=:true)`)
        - `offset`
            - `Real`, optional
            - offset between the hatching lines
            - equivalent to vertical distance between two neighboring lines
            - the default is `0.05`
                - works well for normalized histograms (`Plots.histogram(..., normalize=:true)`)
        - `npoints`
            - `Int`, optional
            - how many points to use for representing each line
            - low resolution might lead to lines not reaching all the way to the border of the bins
            - high resolution might be slow for a lot of bins in the histogram
            - the default is `50`
        - `label`
            - `String`, optional
            - label to give to the hatching
            - implemented as kwarg because proper legend entry does not work yet
            - the default is `""`
                - does not appear in the legend
        - `plot_kwargs`
            - `Vararg{Any}`, optional
            - any other kwargs to pass to `Plots.plot()`, `Plots.plot!()`, `Plots.histogram()`

    Raises
    ------
        - `AssertionError`
            - if `hg` is of the wrong `:seriestype`

    Comments
    --------

"""
function hatch_histogram!(
    hg::Plots.Plot;
    k::T=0.1, offset::T=0.05,
    npoints::Int=50,
    label::String="",
    plot_kwargs...
    ) where {T <: Real}

    #frontmatter checks
    nseries = length(hg.series_list)
    series_list_idx = nseries == 2 ? 1 : length(hg.series_list) - 1
    @assert hg.series_list[series_list_idx][:seriestype] in [:histogram,:shape] "`hg.series_list[$series_list_idx]` has to be generated by `Plots.histogram()`! The passed object has `hg.series_list[$series_list_idx][:seriestype]=$(hg.series_list[series_list_idx][:seriestype])`."

    #definition of a line
    f(x; k=series_list_idx, d=0) = k*x + d

    #extract rectangles that make up the histogram
    x = vcat(hg.series_list[series_list_idx][:x]..., NaN)
    y = vcat(hg.series_list[series_list_idx][:y]..., NaN)
    rectangles = cat(x, y; dims=2)'
    rectangles = reshape(rectangles, (2,6,:))[:,1:end-1,:]
    
    #get lines for every bin
    hatch_x = []    #init hatch-coordinates
    hatch_y = []    #init hatch-coordinates
    for i in axes(rectangles, 3)
        
        #bin bounds
        xi = range(nanminimum(rectangles[1,:,i]), nanmaximum(rectangles[1,:,i]), npoints)
        yi = [nanminimum(rectangles[2,:,i]), nanmaximum(rectangles[2,:,i])]
        
        #get all possible lines
        all_lines = []
        offsets = range(-sign(k)*k, nanmaximum(yi); step=offset)   #offsets to generate multiple stacked lines
        d = -k*nanminimum(sign(k)*xi)*sign(k)  #same as inserting `maximum()` for `k<0` and `minimum` for `k>=0`
        all_lines = [f.(xi; k=k, d=d) .+ os for os in offsets]

        #split lines into x and y values
        #clip all parts that are not contained within a bin
        ly = map(x -> x[(nanminimum(yi) .< x) .& (x .< nanmaximum(yi))], all_lines)
        lx = map(x -> xi[(nanminimum(yi) .< x) .& (x .< nanmaximum(yi))], all_lines)
        
        #add `NaN` to force connecting cosecutive lines
        lx = map(x -> vcat(x..., NaN), lx)
        ly = map(x -> vcat(x..., NaN), ly)

        #add to complete hatch
        hatch_x = vcat(hatch_x, lx...)
        hatch_y = vcat(hatch_y, ly...)
        
    end
    
    
    #plot the hatch
    plot!(hg, hatch_x, hatch_y; label=label, plot_kwargs...)

end
"""
    - function to plot a histogram with custom hatching
    - allows for more artistic freedom compared to `Plots.histogram(fillstyle=...)`

    Parameters
    ----------
        - `plt`
            - `Plots.Plot`
            - panel to plot into
        - `k`
            - `Real`, optional
            - slope of the hatching lines
            - the default is `0.1`
                - works well for normalized histograms (`Plots.histogram(..., normalize=:true)`)
        - `offset`
            - `Real`, optional
            - offset between the hatching lines
            - equivalent to vertical distance between two neighboring lines
            - the default is `0.05`
                - works well for normalized histograms (`Plots.histogram(..., normalize=:true)`)
        - `npoints`
            - `Int`, optional
            - how many points to use for representing each line
            - low resolution might lead to lines not reaching all the way to the border of the bins
            - high resolution might be slow for a lot of bins in the histogram
            - the default is `50`
        - `label`
            - `String`, optional
            - label to give to the hatching
            - implemented as kwarg because proper legend entry does not work yet
            - the default is `""`
                - does not appear in the legend
        - `plot_kwargs`
            - `Vararg{Any}`, optional
            - any other kwargs to pass to `Plots.plot()`, `Plots.histogram()`

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
function hatched_histogram!(
    plt::Plots.Plot, x;
    k::T=0.1, offset::T=0.05,
    npoints::Int=50,
    label::String="",
    plot_kwargs...
    ) where {T <: Real}
    
    #plot normal histogram in background
    histogram!(plt, x; label="", fillalpha=0, plot_kwargs...)

    #add hatching
    hatch_histogram!(plt; k=k, offset=offset, npoints=npoints, label=label, plot_kwargs...)

end
function hatched_histogram(
    x;
    k::T=0.1, offset::T=0.05,
    npoints::Int=50,
    label::String="",
    plot_kwargs...,
    )::Plots.Plot where {T <: Real}
    
    plt = plot()
    hatched_histogram!(plt, x; k=k, offset=offset, npoints=npoints, label=label, plot_kwargs...)
    
    return plt
    
end


#######################################
#parallel coordinates
"""
    - function to smooth along second axis of `x` using 3rd order bezier curves
    - uses additional interpolation techniques to enforce a sigmoid-like morphology

    Parameters
    ----------
        - `x`
            - `AbstractMatrix`
            - matrix of data to be smoothed
        - `yspread`
            - `Real`, optional
            - value to control the random spread assigned to each line
            - used to disentangle lines with the same trajectories
            - the default is `0.02`
                - i.e. `+ 0.02*randn(...)`
        - `dyspread`
            - `Real`, optional
            - value to add control of the random spread w.r.t. the difference in consecutive coordinates
            - will add less spread to lines where the difference is small
                - no influence when difference is `0`
            - the default is `0`
        - `bzfreedom`
            - `Real`, optional
            - parametrized degrees of freedom of the bezier curves used for interpolation
            - the lower, the lower the freedom
            - a value close to `0` will result in nonsmooth curves
                - will have kinks at support points
            - the default is `0.2`

    Raises
    ------
    
    Returns
    -------
        - `xx`
            - `Vector{Vector}`
            - x values of the smoothed version of `x`
        - `yy`
            - `Vector{Vector}`
            - y values of the smoothed version of `x`

    Comments
    --------

"""
function pc_bezier3_smoothing(
    y::AbstractMatrix;
    yspread::Real=0.02,
    dyspread::Real=0,
    bzfreedom::Real=0.2,
    )::Tuple{Vector{Vector}, Vector{Vector}}
    
    #interpolate to ensure that horizontal parts stay (approximately) horizontal
    ##add support point left and right of coordinate #ensures that intersection with coordinate will be horizontal
    xinterp_leftright = Float32.(repeat(axes(y,2), 1, 4))
    xinterp_leftright = reshape(permutedims(xinterp_leftright, (2,1)), :)[2:end-2]  #ignore first and last because that is not shown anyways
    xinterp_leftright[4:4:end] .-= .1
    xinterp_leftright[3:4:end] .+= 0.5  #for midpoint
    xinterp_leftright[2:4:end] .+= .1
    # println("xinterp_leftright: $xinterp_leftright")
    
    yinterp_leftright = repeat(y, 1, 1, 4)  #repeat columns
    yinterp_leftright = reshape(permutedims(yinterp_leftright, (1,3,2)), size(y,1), :)[:,2:end-2]    #ignore first and last because that is not shown anyways
    # println("yinterp_leftright[1,:]: $(yinterp_leftright[1,:])")

    ##add midpoint with noise to disentangle lines with same trajectory
    dy = (y[:,2:end] .- y[:,1:end-1])               #difference between consecutive coordinates
    ymid = (y[:,1:end-1] .+ dy ./ 2)                #midpoint between two consecutive coordinates
    ymid .+= yspread .* (randn(size(ymid,1)) .+ dy) #add noise to disentangle lines with same trajectory
    ymid .*= (1 .+ dyspread*abs.(dy))                      #scale noise based on difference between consecutive coordinates (ensures that dy = 0 results in small-ish)
    yinterp_leftright[:,3:4:size(yinterp_leftright,2),:] .= ymid
    # println("yinterp_leftright[1,:]: $(yinterp_leftright[1,:])")
    
    ##interpolate linearly to ensure curves follow a smoothed sigmoid-like morphology #bezier interpolation will be applied to this variant
    xinterp = range(1,size(y,2);step=bzfreedom)     #controls degrees of freedom the bezier curves will have (the lower the lower the freedom of the bezier curves)
    x_knots = range(1,size(y,2); length=size(yinterp_leftright,2))
    yinterp = mapslices(x -> Interpolations.linear_interpolation(x_knots, x).(xinterp), yinterp_leftright; dims=2)

    ##get bezier interpolated lines (i.e. smoothing)
    b3i = Bezier3Interp.Bz3Interp() #instantiate
    xyinterp = dropdims(mapslices(x -> Bezier3Interp.fit(b3i, [xinterp x]'), yinterp; dims=2); dims=2)
    xyinterp = map(x -> Bezier3Interp.transform(x, size(yinterp,2)*10), xyinterp)
    xx = map(x -> x[1,:], xyinterp) #extract x coordinate
    yy = map(x -> x[2,:], xyinterp) #extract y coordinate
    
    return xx, yy
end
"""
    - function to interpolate second axis of `x` by placing scaled sigmoids between two adjacent indices
    - each sigmoid will start at `x[:,i-1]` and end at `x[:,i]` with gradual in/decrease between the values
    - for each sample (first axis) in `x` a unique slope will be generated to make sure curves with identical trajectories are disentangled

    Parameters
    ----------
        - `x`
            - `Matrix`
            - input data to be interpolated
        - `res`
            - `Int`, optional
            - resolution of each sigmoid
            - the final curve will have a second axis with length `res*(size(x,2)-1)`
            - the default is `20`
        - `slopes_min`
            - `Real`, optional
            - minimum slope to use for the sigmoids
            - if you want all curves to have the same slope just set `slopes_min` equal to `slopes_max`
            - the default is `1`
        - `slopes_max`
            - `Real`, optional
            - maximum slope to use for the sigmoids
            - if you want all curves to have the same slope just set `slopes_min` equal to `slopes_max`
            - the default is `2`

    Raises
    ------

    Returns
    -------
        - `xx`
            - `AbstractRange`
            - xvalues used to generate the interpolated version of `y`
            - will have `res*(size(x,2)-1)` values ranging from `1` to `size(x,2)`
        - `yy`
            - `Matrix`
            - interpolated version of `x`

    Comments
    --------
        - usually used for a parallel-coordinates plot
"""
function pc_sigmoid_itp(
    x::Matrix;
    res::Int=20,
    slopes_min::Real=1, slopes_max::Real=2,
    )::Tuple{AbstractRange,Matrix}

    #parameters
    nsampels, ncoords = size(x)
    
    slopes = reshape(range(slopes_min,slopes_max,nsampels), :, 1, 1)    #ensure every line has a distinct slope (disentangle lines)
    dy = x[:,2:end] - x[:,1:end-1]                                      #get difference of 2 consecutive coordinates (spanned range)
    sm = mapslices(x -> sigmoid.(range(-10,10,res), x), slopes; dims=3) #generate sigmoids with distinct slopes to use for each line
    yy = sm .* dy .+ x[:,1:end-1]                                       #rescale sigmoids to have correct range (dy) and offset #inverse minmaxscaler
    yy = reshape(permutedims(yy, (1,3,2)), size(x,1), :)                #reshape for plotting
    xx = range(1,ncoords,res*(ncoords-1))                               #xvalues for plotting
    
    return xx, yy
end
"""
    - function to create a Prallel-Coordinate plot
    - inspired by the Weights&Biases' Parallel-Coordinates plot 
        - https://docs.wandb.ai/guides/app/features/panels/parallel-coordinates (last access: 15.05.2023)

    Parameters
    ----------
        - `plt`
            - `Plots.Plot`
            - panel to plot into
        - `df`
            - `DataFrame`
            - the dataframe to visualize
            - columns are interpreted as coordinates
            - rows are interpreted samples
        - `res`
            - `Int`, optional
            - resolution of each sigmoid used to interpolate the coordinates
            - each final line will have a resolution of `res*(ncols(df)-1)` datapoints
            - the default is `20`
        - `slopes_min`
            - `Real`, optional
            - minimum slope to use for the sigmoids in the interpolation
                - sigmoids of any distinct sample/line will have the same slope
            - if you want all curves to have the same slope just set `slopes_min` equal to `slopes_max`
            - the default is `1`
        - `slopes_max`
            - `Real`, optional
            - maximum slope to use for the sigmoids in the interpolation
                - sigmoids of any distinct sample/line will have the same slope
            - if you want all curves to have the same slope just set `slopes_min` equal to `slopes_max`
            - the default is `2`
        - `cmap`
            - `Symbol`, `ColorPalette`, optional
            - colormap to use for plotting the lines
            - used to color based on the last column in `df`
            - if `ColorPalette`
                - has to have same length as `nrow(df)`   
                - for example `palette(:plasma, nrow(df))`
            - the default is `:plasma`
        - `n_yticks`
            - `Int`, optional
            - how many ticks to use on each coordinate axis
            - will set the exact number of ticks if the number of unique values in a coordinate <= `n_yticks`
            - the default is `5`
        - `ytickfontsize`
            - `Int`, optional
            - fontsize to use for the ticklabels on the coordinate axes
            - similar to the standard`ytickfontsize` kwarg except applied to all y axes
            - the default is `nothing`
                - will be set to `Int(round(2/3*Plots.default(:tickfontsize)))`
                - i.e. 2/3 of the fontsize used for ticklabels in the current theme
        - `yrotation`
            - `AbstractFloat`, optional
            - rotation to apply to the yticklabels
            - the default is `-40.`
        - `ytickcolor`
            - `RGB`, `Symbol`, optional
            - colors to use for drawing the ticks of the coordinate axes
            - if `Symbol`
                - has to be a valid color
            - the default is `nothing`
                - will be set to `Plots.default(:fgguide)`
                - i.e. the color applied to figure-guides in the current theme
        - `yticklabelcolor`
            - `RGB`, `Symbol`, optional
            - colors to use for drawing the ticklabels of the coordinate axes
            - if `Symbol`
                - has to be a valid color
            - the default is `nothing`
                - will be set to `Plots.default(:fgtext)`
                - i.e the color applied to text in the figure in the current theme
        -`bg`
            - `RGBA`, `Symbol`, optional
            - color to use for drawing the background of the ticklabels of the coordinate axes
            - if `Symbol`
                - has to be a valid color
            - the default is `nothing`
                - will be set to `RGBA(RGB(Plots.default(:bg)), 0.8)`
                - i.e. a slightly transparent version of the color used for the figure background in the current theme
        - `ytickalpha`
            - `AbstractFloat`, optional
            - transparency to apply to the ticks of the coordinate axes
            - the default is `1.0`
        - `yticksize`
            - `AbstractFloat`, optional
            - scaling factor to increase the length of the ticks of the coordinate axes
            - the actual size is computed according to `yticksize * 3*sz[1]/sz[2]`
                - where `sz = plt.attr[:size]`
            - the default is `1.0`
        - `plot_kwargs`
            - `Vararg`, optional
            - additional kwargs to pass to `Plots.plot!()` when plotting the lines

    Raises
    ------

    Returns
    -------
        - `plt`
            - `Plots.Plot`
            - created panel

    Comments
    --------
        - hovering over a datapoint shows the coordinates that lead to the value on the last y-axis
            - only works in `plotlyjs`
        - in case you want to use `parallel_coords()` or `parallel_coords!()` with `pgfplotsx`
            - make sure to add the following lines to your script
                - `using PGFPlotsX`
                - `PGFPlotsX.CUSTOM_PREAMBLE = ["\\usepackage{pmboxdraw}", "\\usepackage{graphicx}", "\\newcommand{\\blockfull}{\\scalebox{.4}[1]{\\textblock}}"]`


"""
function parallel_coords!(plt::Plots.Plot,
    df::DataFrame;
    res::Int=20,
    slopes_min::Real=1, slopes_max::Real=2,
    cmap::Union{Symbol,ColorPalette}=:plasma,
    #yticks options
    n_yticks::Int=5,
    ytickfontsize::Union{Int,Nothing}=nothing, yrotation::AbstractFloat=-40.,
    ytickcolor::Union{RGB,Symbol,Nothing}=nothing, yticklabelcolor::Union{RGB,Symbol,Nothing}=nothing, bg::Union{RGBA,Symbol,Nothing}=nothing,
    ytickalpha::Union{AbstractFloat,Nothing}=1.,
    yticksize::AbstractFloat=1.,
    #additional kwargs
    plot_kwargs...    
    )

    #default parameters
    ##yticks
    ytickfontsize       = isnothing(ytickfontsize)   ? Int(round(2/3*Plots.default(:tickfontsize))) : ytickfontsize
    ytickalpha          = isnothing(ytickalpha)      ? Plots.default(:gridalpha) : ytickalpha
    ytickcolor          = isnothing(ytickcolor)      ? RGB(Plots.default(:fgguide))  : ytickcolor
    yticklabelcolor     = isnothing(yticklabelcolor) ? RGB(Plots.default(:fgtext))   : yticklabelcolor
    bg                  = isnothing(bg) ? RGBA(RGB(Plots.default(:bg)), 0.8) : bg
    text_kwargs = (     #kwargs to pass to `text()` #affects style of the yticklabels
        halign=:left, valign=:left,
        pointsize=ytickfontsize,
        rotation=yrotation,
    )
    #transform string columns to numeric mapping #rescale for plotting
    df_num = select(df, :, names(df, AbstractString) .=> x -> levelcode.(categorical(x)), renamecols=false)
    df_scaled = mapcols(minmaxscaler, df_num)
    
    #init  colormapping
    if isa(cmap, Symbol)
        cmap = palette(cmap, nrow(df_scaled))
    else
        @assert length(cmap) ==  nrow(df_scaled) "`cmap` has to be a `ColorPalette` of length equal to `nrow(df)`"
    end  


    #convert to matrix for plotting
    mat_scaled = Matrix(df_scaled)
    # println("size(mat_scaled): $(size(mat_scaled))")
    
    # #smooth using cubic bezier interpolation
    # xx, yy = pc_bezier3_smoothing(mat_scaled; yspread=yspread, dyspread=dyspread, bzfreedom=bzfreedom)
    
    #smooth using sigmoid interpolation
    xx, yy = pc_sigmoid_itp(mat_scaled; res=res, slopes_min=slopes_min, slopes_max=slopes_max)

    # println("size(xx): $(size(xx))")
    # println("size(yy): $(size(yy))")
    
    # display(mat_scaled[[1,end],:])
    # p = plot(xx, yy[[1,end],:]')
    # vline!()
    # # p = plot(sigmoid.(xx_, .5))
    # # plot!(p, sigmoid.(xx_, 1))
    # # plot!(p, sigmoid.(xx_, 2))
    # display(p)
    
    #plot lines
    show_on_hover = reshape(map(row -> "($(join(row[1:end-1], ","))) => $(row[end])", eachrow(df)), 1,:)
    plot_kwargs = Plots.backend_name() == :plotlyjs ? [plot_kwargs..., :hover => show_on_hover] : plot_kwargs
    plot!(plt,
        xx, yy';
        color_palette=[cmap...],
        label="",
        xlims=(1,ncol(df_scaled)), ylims=(-.1,1.05),   #10% margin yticklabels and xticklabels
        xticks=(1:ncol(df_scaled), names(df_scaled)), yticks=false,       #no yticks because custom yticks added
        xrotation=90,
        # hover=show_on_hover,    #only works with `plotlyjs`
        grid=true, minorgrid=false,
        legend=false,
        plot_kwargs...,
    )
        
    begin   #add yticks including labels
        yticks_x = []                           #x-coordinates of the ticks to be added
        yticks_y = []                           #y-coordinates of the ticks to be added
        yticklabels = Matrix{Any}(undef, 0, 3)  #init list of labels
        ytickboxes  = Matrix{Any}(undef, 0, 3)  #init list of labels
        for (i, (c, cs)) in enumerate(zip(eachcol(df), eachcol(df_scaled)))        
            
            #yvalues differ depending on column type
            #NOTE: `█` is a filled space => can be used to define boxes around text
                #NOTE: it is not possible to use `█` with `pgfplotsx()` without definint a custom preamble!
            if eltype(c) <: Number  #continuous range for numbers
                if size(unique(c),1) <= n_yticks         #show exact ticks if numeric and less unique values than `n_yticks`
                    xcoords = repeat([i], size(unique(c),1))
                    ycoords = unique(cs)
                    vals = unique(c)
                else                                        #plot some continuous range
                    xcoords = repeat([i], n_yticks)
                    ycoords = range(0,1, n_yticks)
                    vals = range(minimum(c), maximum(c), n_yticks)
                end
                vals = map(x -> @sprintf(" %g", x), vals)    #map to strings
                yl = hcat(xcoords, ycoords, map(x -> text(x, color=yticklabelcolor, text_kwargs...), vals))
                yb = hcat(xcoords, ycoords, map(x -> text("█"^(length(x)), color=bg, text_kwargs...), vals))
                push!(yticks_x, xcoords)
                push!(yticks_y, ycoords)
                yticklabels = vcat(yticklabels, yl)
                ytickboxes  = vcat(ytickboxes, yb)
            else    #all discrete values for strings/categorical
                xcoords = repeat([i], size(unique(cs),1))
                ycoords = unique(cs)
                yl = hcat(xcoords, ycoords, map(x -> text("$x", color=yticklabelcolor, text_kwargs...), unique(c) ))
                yb = hcat(xcoords, ycoords, map(x -> text("█"^(length("$x")), color=bg, text_kwargs...), unique(c)))
                push!(yticks_x, xcoords)
                push!(yticks_y, ycoords)
                yticklabels = vcat(yticklabels, yl)
                ytickboxes  = vcat(ytickboxes, yb)
            end
        end
        
        #plot yticks
        sz = plt.attr[:size]    #plot size
        ts = yticksize * 3*sz[1]/sz[2]  #compute tiksize based on aspect ratio and custom scaling parameter
        scatter!(plt, yticks_x, yticks_y;   #actual ticks
            markershape=:hline,
            color=ytickcolor,
            markercolor=ytickcolor, markeralpha=ytickalpha,
            markersize=ts,
            label="",
        )
        annotate!(plt, ytickboxes[:,1], ytickboxes[:,2], ytickboxes[:, 3])
        annotate!(plt, yticklabels[:,1], yticklabels[:,2], yticklabels[:,3])
    end

    begin   #add histogram of scores
        #TODO
    end

end
function parallel_coords(
    df::DataFrame;
    res::Int=20,
    slopes_min::Real=1, slopes_max::Real=2,
    cmap::Union{Symbol,ColorPalette}=:plasma,
    #yticks options
    n_yticks::Int=5,
    ytickfontsize::Union{Int,Nothing}=nothing, yrotation::AbstractFloat=-40.,
    ytickcolor::Union{RGB,Symbol,Nothing}=nothing, yticklabelcolor::Union{RGB,Symbol,Nothing}=nothing, bg::Union{RGBA,Symbol,Nothing}=nothing,
    ytickalpha::Union{AbstractFloat,Nothing}=1.,
    yticksize::AbstractFloat=1.,
    #additional kwargs
    plot_kwargs...  
    )::Plots.Plot
    
    plt = plot()
    parallel_coords!(plt,
        df;
        res=res,
        slopes_min=slopes_min, slopes_max=slopes_max,
        cmap=cmap,
        n_yticks=n_yticks,
        ytickfontsize=ytickfontsize, yrotation=yrotation,
        ytickcolor=ytickcolor, yticklabelcolor=yticklabelcolor, bg=bg,
        ytickalpha=ytickalpha,
        yticksize=yticksize,
        plot_kwargs...,
    )
    
    return plt

end


end #module