"""
    - module implementing sigma-clipping

    
    Structs
    -------
        - `SigmaClipping`
        - `SigmaClippingResult`
    
    Functions
    ---------
        - `fit_bng()`
        - `fit_poly()`
        - `fit()`
        - `transform()`

    Extended Functions
    ------------------
        - `Plots.plot()`
        - `Plots.plot!()`
    
    Dependencies
    ------------
        - `Binning.jl` (this package)
        - `FormattingUtils.jl` (this package)
        - `Interpolations`
        - `LaTeXStrings`
        - `NaNStatistics`
        - `Plots`
        - `Polynomials`

    Examples
    --------
        - see [../src_demos/SigmaClipping_demo.jl](../src_demos/SigmaClipping_demo.jl)
"""

#%%imports
using Interpolations
using LaTeXStrings
using NaNStatistics
using Plots
using Polynomials

#import for extending
import Plots: plot, plot!

#intradependencies
include(joinpath(@__DIR__, "./Binning.jl"))
include(joinpath(@__DIR__, "./FormattingUtils.jl"))

#%%definitions

#######################################
#sigma-clipping
"""
    - struct defining the sigma-clipping transformer
    
    Fields
    ------
        - `sigma_l`
            - `Real`, optional
            - multiplier for defining the upper bound
            - upper bound is defined by `r .+ sigma_l .* sigma`
                - `r` is hereby the
                    - binned curve if `method=:binning`
                    - the polynomial fit if `method=:poly`
                - `sigma` is hereby the
                    - standard deviation of the bin if `method=:binning`
                    - the global standard deviation if `method=:poly`
            - the default is `1.5`
        - `sigma_u`
            - `Real`, optional
            - multiplier for defining the lower bound
            - lower bound is defined by `sigma_l .* sigma`
                - `sigma` is hereby the
                    - standard deviation of the bin if `method=:binning`
                - the global standard deviation if `method=:poly`
            - the default is `1.5`
        - `max_iter`
            - `Int`, optional
            - maximum number of iterations to consecutively apply sigma-clipping on the data
            - the default is 1
        - `sigma_l_decay`
            - `Real`, optional
            - decay rate of `sigma_l`
            - `sigma_l` of iteration `k` is computed as `sigma_l * sigma_l_decay^k`
            - the default is `1`
                - no decay
        - `sigma_u_decay`
            - `Real`, optional
            - decay rate of `sigma_u`
            - `sigma_l` of iteration `k` is computed as `sigma_u * sigma_u_decay^k`
            - the default is `1`
                - no decay
        - `method`
            - `Symbol`, optional
            - method to use for `SigmaClipping`
            - has to be one of `:binning`, `:poly`

    Methods
    -------
        - `fit_bng()`
        - `fit_poly()`    
        - `fit()`
        - `transform()`
        - `Plots.plot()`
        - `Plots.plot!()`

    Related Structs
    ---------------
        - `SigmaClippingResult`
    
    Comments
    --------
        - `SigmaClipping` is not designed to be able to deal with `NaN`
            - clean your data from `NaN` before applying!

"""
struct SigmaClipping
    sigma_l::Real
    sigma_u::Real
    max_iter::Int
    sigma_l_decay::Real
    sigma_u_decay::Real
    method::Symbol

    #inner constructor (to allow default values)
    function SigmaClipping(
        sigma_l::Real=1.5, sigma_u::Real=1.5;
        max_iter::Int=1,
        sigma_l_decay::Real=1., sigma_u_decay::Real=1.,
        method::Symbol=:binning,
        )

        @assert method in [:binning,:poly] ArgumentError("`method` has to be one of `:binning`, `:poly` but is `$method`!")

        new(sigma_l, sigma_u, max_iter, sigma_l_decay, sigma_u_decay, method)
    end
end

"""
    - struct to store result of `SigmaClipping`

    Fields
    ------
        - `clip_mask`
            - `Vector{Bool}`   
            - mask resulting from the sigma-clipping
            - has `true` whereever the datapoint shall be retained
            - has `false` whereever the datapoint shall be clipped
        - `lb`
            - `Vector{AbstractFloat}`
            - lower bound used in the final iteration
        - `ub`
            - `Vector{AbstractFloat}`
            - upper bound used in the final iteration
            - `Vector{Real}`
        - `x_repr`
            - `Vector{AbstractFloat}`
            - x-values of the representative curve used in the final iteration 
        - `y_repr`
            - `Vector{AbstractFloat}`
            - y-values of the representative curve used in the final iteration 
"""
struct SigmaClippingResult
    clip_mask::Vector{Bool}
    lb::Vector{AbstractFloat}
    ub::Vector{AbstractFloat}
    x_repr::Vector{AbstractFloat}
    y_repr::Vector{AbstractFloat}
end

"""
    - function to fit `SigmaClipping` to data using `Binning` for the representative function
    - i.e. exectues sigma-clipping given the parameters of `SigmaClipping`
    
    Parameters
    ----------
        - `sc`
            - `SigmaClipping`
            - struct containing the hyperparameters of the model
        - `x`
            - `Vector{Real}`
            - x-values of the dataseries to be clipped
        - `y`
            - `Vector{Real}`
            - y-values of the dataseries to be clipped
        - `generate_bins`
            - `Function`, optional
            - function to be used to generate the bins for `Binning` in each iteration `k`
            - has to take at least 2 args and 1 kwarg
                - `x[clip_mask]`
                    - x-values of the current iterations
                - `generate_bins_args`
                - `generate_bins_kwargs`
            - will be called as `generate_bins(x[clip_mask], generate_bins_args...; generate_bins_kwargs...)`
            - the default is `nothing`
                - will be set to `Binning.generate_bins_pts()`
        - `y_start`
            - `Real`, optional
            - value to use for the first datapoint in `y_repr`
                - ensures that `y_repr(minimum(x))` exists
                - necessary because `Binning` will not cover the complete interpolation range
            - the default is `nothing`
                - will be set to `y_repr[1]` for every iteration
        - `y_end`
            - `Real`, optional
            - value to use for the last datapoint in `y_repr`
                - ensures that `y_repr(maximum(x))` exists
                - necessary because `Binning` will not cover the complete interpolation range
            - the default is `nothing`
                - will be set to `y_repr[end]` for every iteration
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`
        - `generate_bins_args`
            - `Tuple`, optional
            - additional args to pass to `generate_bins`
            - the default is `nothing`
                - will be set to `()`
                - empty `Tuple`
        - `generate_bins_kwargs`            
            - `NamedTuple`, optional
            - additional kwargs to pass to `generate_bins`
            - the default is `nothing`
                - will be set to `(verbose=verbose-3,)`
        - `binning_kwargs`
            - `NamedTuple`, optional
            - additional kwargs to pass to `Binning`
            - the default is `nothing`
                - will be set to `()`
    
    Raises
    ------

    Returns
    -------
        - `scr`
            - `SigmaClippingResult`
            - struct containing computed quantities

    Comments
    --------
"""
function fit_bng(
    sc::SigmaClipping,
    x::Vector{T}, y::Vector{T};
    generate_bins::Union{Function,Nothing}=nothing,
    y_start::Union{Real,Nothing}=nothing, y_end::Union{Real,Nothing}=nothing,
    verbose::Int=0,
    generate_bins_args::Union{Tuple,Nothing}=nothing,
    generate_bins_kwargs::Union{NamedTuple,Nothing}=nothing,
    binning_kwargs::Union{NamedTuple,Nothing}=nothing
    ):: SigmaClippingResult where {T<:Real}

    #default parameter
    generate_bins           = isnothing(generate_bins) ? generate_bins_pts : generate_bins
    generate_bins_args      = isnothing(generate_bins_args)   ? () : generate_bins_args
    generate_bins_kwargs    = isnothing(generate_bins_kwargs) ? (verbose=verbose-3,) : generate_bins_kwargs
    binning_kwargs          = isnothing(binning_kwargs)       ? () : binning_kwargs

    #init outputs
    clip_mask = ones(Bool, size(x)) #current clip-mask
    lb = [NaN]
    ub = [NaN]
    x_repr = [NaN]
    y_repr = [NaN]
    for k in 1:sc.max_iter
        
        printlog(
            "Iteration $k/$(sc.max_iter)\n";
            context="fit(SigmaClipping)",
            type=:INFO,
            level=0,
            verbose=verbose,
        )

        #init `Binning` for current iteration
        bins = generate_bins(x[clip_mask], generate_bins_args...; generate_bins_kwargs...)
        bng = Binning(bins; binning_kwargs...)
        
        #get repr curve
        br = fit(bng, x[clip_mask], y[clip_mask]; verbose=verbose-2)
        x_repr, y_repr, x_std, y_std = transform(br)
        
        #set `y_start` and `y_end` for the iteration accordingly
        y_start_k = isnothing(y_start) ? copy(y_repr[1])     : y_start
        y_end_k   = isnothing(y_end)   ? copy(y_repr[end])   : y_end
        
        #add minimum and maximum to ensure correct interpolation range
        x_repr = vcat(x[1], x_repr..., x[end])
        y_repr = vcat(y_start_k, y_repr..., y_end_k)
        y_std  = vcat(y_std[1], y_std..., y_std[end])
        
        #sort to ensure correct formattting for interpolation
        sidxs = sortperm(x_repr)
        x_repr  = x_repr[sidxs]
        y_repr  = y_repr[sidxs]
        y_std   = y_std[sidxs]
        
        #remove `NaN`
        nanmask = isfinite.(x_repr) .& isfinite.(y_repr) .& isfinite.(y_std)
        x_repr  = x_repr[nanmask]
        y_repr  = y_repr[nanmask]
        y_std   = y_std[nanmask]

        #get bounds
        sigma_l_k = sc.sigma_l * sc.sigma_l_decay ^(k-1)
        sigma_u_k = sc.sigma_u * sc.sigma_u_decay ^(k-1)
        lb = y_repr .- (sigma_l_k .* y_std)
        ub = y_repr .+ (sigma_u_k .* y_std)
        
        Interpolations.deduplicate_knots!(x_repr)   #deduplicate to supress warning from `Interpolations`
        
        lb_itp = Interpolations.linear_interpolation(x_repr, lb; extrapolation_bc=NaN)
        ub_itp = Interpolations.linear_interpolation(x_repr, ub; extrapolation_bc=NaN)
        
        #update clip_mask
        clip_mask = clip_mask .&& ((lb_itp(x) .< y) .& (y .< ub_itp(x)))

    end

    return SigmaClippingResult(clip_mask, lb, ub, x_repr, y_repr)

end
"""
    - function to fit `SigmaClipping` to data using polynomials for the representative function
    - i.e. exectues sigma-clipping given the parameters of `SigmaClipping`
    
    Parameters
    ----------
        - `sc`
            - `SigmaClipping`
            - struct containing the hyperparameters of the model
        - `x`
            - `Vector{Real}`
            - x-values of the dataseries to be clipped
        - `y`
            - `Vector{Real}`
            - y-values of the dataseries to be clipped
        - `p_deg`
            - `Int`, optional
            - polynomial degree of the representative function
            - will fit polynomail of degree `p_deg` to the data
            - the default is `1`
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`
    
    Raises
    ------

    Returns
    -------
        - `scr`
            - `SigmaClippingResult`
            - struct containing computed quantities

    Comments
    --------
"""
function fit_poly(
    sc::SigmaClipping,
    x::Vector{T}, y::Vector{T};
    p_deg::Int=1,
    verbose::Int=0,
    ):: SigmaClippingResult where {T<:Real}

    #init outputs
    clip_mask = ones(Bool, size(x)) #current clip-mask
    x_repr = copy(x)
    y_repr = [NaN]
    lb = [NaN]
    ub = [NaN]
    for k in 1:sc.max_iter        
        printlog(
            "Iteration $k/$(sc.max_iter)\n";
            context="fit(SigmaClipping)",
            type=:INFO,
            level=0,
            verbose=verbose,
        )

        #get standard deviation for iteration `k`
        std_k = nanstd(y[clip_mask])
        
        #get repr curve
        p = Polynomials.fit(x[clip_mask], y[clip_mask], p_deg)
        y_repr = p.(x)

        #get bounds
        sigma_l_k = sc.sigma_l * sc.sigma_l_decay ^(k-1)
        sigma_u_k = sc.sigma_u * sc.sigma_u_decay ^(k-1)        
        lb = y_repr .- (sigma_l_k .* std_k)
        ub = y_repr .+ (sigma_u_k .* std_k)
        
        #update clip_mask
        clip_mask = clip_mask .&& ((lb .< y) .& (y .< ub))

    end

    return SigmaClippingResult(clip_mask, lb, ub, x_repr, y_repr)    

end
"""
    - function to fit `SigmaClipping` to data using the `method` specified in `SigmaClipping`
    - i.e. exectues sigma-clipping given the parameters of `SigmaClipping`
    
    Parameters
    ----------
        - `sc`
            - `SigmaClipping`
            - struct containing the hyperparameters of the model
        - `x`
            - `Vector{Real}`
            - x-values of the dataseries to be clipped
        - `y`
            - `Vector{Real}`
            - y-values of the dataseries to be clipped
        - `generate_bins`
            - `Function`, optional
            - function to be used to generate the bins for `Binning` in each iteration `k`
            - has to take at least 2 args and 1 kwarg
                - `x[clip_mask]`
                    - x-values of the current iterations
                - `generate_bins_args`
                - `generate_bins_kwargs`
            - will be called as `generate_bins(x[clip_mask], generate_bins_args...; generate_bins_kwargs...)`
            - the default is `nothing`
                - will be set to `Binning.generate_bins_pts()`
        - `y_start`
            - `AbstractFloat`, optional
            - value to use for the first datapoint in `y_repr`
                - ensures that `y_repr(minimum(x))` exists
                - necessary because `Binning` will not cover the complete interpolation range
            - the default is `nothing`
                - will be set to `y[1]`
        - `y_end`
            - `AbstractFloat`, optional
            - value to use for the last datapoint in `y_repr`
                - ensures that `y_repr(maximum(x))` exists
                - necessary because `Binning` will not cover the complete interpolation range
            - the default is `nothing`
                - will be set to `y[end]`
        - `p_deg`
            - `Int`, optional
            - polynomial degree of the representative function
            - will fit polynomail of degree `p_deg` to the data
            - the default is `1`                
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`
        - `generate_bins_args`
            - `NamedTuple`, optional
            - additional args to pass to `generate_bins`
            - the default is `nothing`
                - will be set to `()`
                - empty `Tuple`
        - `generate_bins_kwargs`            
            - `NamedTuple`, optional
            - additional kwargs to pass to `generate_bins`
            - the default is `nothing`
                - will be set to `(:verbose=>verbose-3)`
    
    Raises
    ------

    Returns
    -------
        - `scr`
            - `SigmaClippingResult`
            - struct containing computed quantities

    Comments
    --------
"""
function fit(
    sc::SigmaClipping,
    x::Vector{T}, y::Vector{T};
    #binning kwargs
    generate_bins::Union{Function,Nothing}=nothing,
    y_start::Union{AbstractFloat,Nothing}=nothing, y_end::Union{AbstractFloat,Nothing}=nothing,
    #poly kwargs
    p_deg::Int=Int(1),
    #shared kwargs
    verbose::Int=0,
    #additional args and kwargs
    generate_bins_args::Union{Tuple,Nothing}=nothing,
    generate_bins_kwargs::Union{NamedTuple,Nothing}=nothing,
    binning_kwargs::Union{NamedTuple,Nothing}=nothing,
    )::SigmaClippingResult where {T<:Real}

    #preliminary checks
    @assert all(isfinite.(x)) && all(isfinite.(y)) "`SigmaClipping` is not designed to be able to deal with `NaN`. Clean your data from `NaN` before applying!"

    # println("METHOD ", sc.method)
    if sc.method == :binning
        scr = fit_bng(sc, x, y;
            generate_bins=generate_bins,
            y_start=y_start, y_end=y_end,
            verbose=verbose,
            generate_bins_args=generate_bins_args,
            generate_bins_kwargs=generate_bins_kwargs,
            binning_kwargs=binning_kwargs,
        )        
    elseif sc.method == :poly
        scr = fit_poly(sc, x, y;
            p_deg=p_deg,
            verbose=2,
        )
    else
        throw(ArgumentError("`method` has to be one of `:binning`, `:poly` but is `$method`!"))
    end

    return scr

end

"""
    - function to transform data based on parameters stored in `SigmaClippingResult`
    - i.e. parameters resulting from sigma-clipping
    
    Parameters
    ----------
        - `scr`
            - `SigmaClippingResult`
            - struct containing parameters inferred by `SigmaClipping`
        - `x`
            - `Vector`
            - x-values of the dataseries to be transformed
            - `scr.clip_mask` will be applied
        - `y`
            - `Vector`
            - y-values of the dataseries to be transformed
            - `scr.clip_mask` will be applied
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`
    
    Raises
    ------

    Returns
    -------
        - `x_clipped`
            - `AbstractArray`
            - clipped version of `x`
        - `y_clipped`
            - `AbstractArray`
            - clipped version of `y`

    Comments
    --------
"""
function transform(
    scr::SigmaClippingResult,
    x::Union{Vector,Nothing}, y::Union{Vector,Nothing};
    verbose::Int=0,
    )::NTuple{2, AbstractArray}

    x_clipped = x[scr.clip_mask]
    y_clipped = y[scr.clip_mask]

    return x_clipped, y_clipped

end


"""
    - extensions to `Plots.plot!()` and `Plots.plot()`

    Parameters
    ----------
        - `plt`
            - `Plots.Plot`
            - panel to plot into    
        - `scr`
            - `SigmaClippingResult`
            - struct containing the result from `fit(SigmaClipping)` that shall be visualized
        - `x`
            - `Vector`, optional
            - original x-values corresponding to `br`
            - the default is `nothing`
                - ignored
        - `y`
            - `Vector`, optional
            - original y-values corresponding to `br`
            - the default is `nothing`
                - ignored
        - `sc`
            - `SigmaClipping`, optional
            - `SigmaClipping` instance used to achive `scr`
            - the default is `nothing`
                - ignored
        - `show_clipped`
            - `Bool`, optional
            - whether to show the clipped datapoints as well
            - the deafault is `false`
                - only the retained datapoints are displayed

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
    scr::SigmaClippingResult;
    x::Union{Vector,Nothing}=nothing, y::Union{Vector,Nothing}=nothing,
    sc::Union{SigmaClipping,Nothing}=nothing,
    show_clipped::Bool=false,
    )

    #current theme defaults
    lw = Plots.default(:linewidth)  #linewidth of current theme
    bg = Plots.default(:bg)         #background

    #default parameters
    l_lb = isnothing(sc) ? "Lower Bound" : L"$\vec\mu-%$(sc.sigma_l_decay)^{%$(sc.max_iter)}%$(sc.sigma_l)\vec\sigma$"
    l_ub = isnothing(sc) ? "Upper Bound" : L"$\vec\mu+%$(sc.sigma_u_decay)^{%$(sc.max_iter)}%$(sc.sigma_u)\vec\sigma$"

    #input scatter (only if `x` and ` y` were passed)
    if ~isnothing(x) && ~isnothing(y)
        scatter!(plt, x[scr.clip_mask], y[scr.clip_mask]; label="Retained")
        show_clipped ? scatter!(plt, x[.~scr.clip_mask], y[.~scr.clip_mask]; label="Clipped") : false
    end
    
    #representative curve
    plot!(plt,  #outline
        scr.x_repr, scr.y_repr,
        ls=:solid, lw=2.5*lw, color=bg,
        label=""
    )
    plot!(plt,
        scr.x_repr, scr.y_repr,
        ls=:solid, 
        label="Representative Curve"
    )

    #boundaries
    plot!(scr.x_repr, scr.lb; label="",   ls=:solid, lw=2.5*lw, color=bg,)
    plot!(scr.x_repr, scr.lb; label=l_lb, ls=:dash)
    plot!(scr.x_repr, scr.ub; label="",   ls=:solid, lw=2.5*lw, color=bg,)
    plot!(scr.x_repr, scr.ub; label=l_ub, ls=:dashdot)

end
function Plots.plot(
    scr::SigmaClippingResult;
    x::Union{Vector,Nothing}=nothing, y::Union{Vector,Nothing}=nothing,
    sc::Union{SigmaClipping,Nothing}=nothing,
    show_clipped::Bool=false,
    )::Plots.Plot
    plt = plot(;xlabel=L"$x$", ylabel=L"$y$", title="Sigma-Clipping")
    plot!(plt, scr; x=x, y=y, sc=sc, show_clipped=show_clipped)
    return plt
end

