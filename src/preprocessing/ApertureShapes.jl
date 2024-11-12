
"""
    - module to define custom aperture masks

    Constants
    ---------

    Functions
    ---------
        - `lorentz()`
        - `lp_aperture()`
        - `rect_aperture()`
        - `gauss_aperture()`
        - `lorentz_aperture()`
        - `plot_aperture!()`
        - `plot_aperture()`
        - `print_aperture()`

    Structs
    -------
        - `ApertureTemplate`
        - `ApertureMask`

    Dependencies
    ------------
        - `LinearAlgebra`
        - `NaNStatistics`
        - `Plots`
        - `Polynomials`
        - `Printf`

    Comments
    --------

    Examples
    --------
        - see[../src_demos/ApertureShapes_demo.jl](../src_demos/ApertureShapes_demo.jl)
"""
module ApertureShapes

#%%imports
using LinearAlgebra
using NaNStatistics
using Plots
using Polynomials
using Printf

#exports
export ApertureTemplate
export ApertureMask
export lorentz
export lp_aperture
export rect_aperture
export gauss_aperture
export lorentz_aperture
export plot_aperture
export plot_aperture!
export print_aperture

#intradependencies

#import for extending

#%%helper functions
"""
    - funtion computing a lorentzian profile
    - \$L = \frac{1}{1+x^2}\$
        with \$x = \frac{p - p_0}{\frac{fwhm}{2}}\$

    Parameters
    ----------
        - `p`
            - `Real`
            - independent variable
        - `p0`
            - `Real`, optional
            -  midpoint of the profile
            - the default is `0.`
        - `fwhm`
            - `Real`, optional
            - full width half maximum
                - defines spread of the profile
            - the default is `1.`

    Raises
    ------

    Returns
    -------

    Comments
    --------
"""
function lorentz(
    p::Real;
    p0::Real=0., fwhm::Real=1.,
    )::Real

    x = (p - p0)/(fwhm/2)
    l = 1/(1 + x^2)
    
    return l

end

#%%aperture templates
"""

    - struct to define a aperture-mask constructor
    - will construct an aperture-mask frame of size `(y_size,x_size)`

    Fields
    ------
        - `x_size`
            - `Int`
            - size of the aperture mask in x-direction in pixels
            - will generate frame with size `x_size` in x-direction
        - `y_size`
            - `Int`
            - size of the aperture mask in y-direction in pixels
            - will generate frame with size `y_size` in y-direction
        - `x`
            - `Number`, optional
            - position of the aperture in x-direction
            - the default is `0`
                - centered
        - `y`
            - `Number`, optional
            - position of the aperture in y-direction
            - the default is `0`
                - centered
        - `npixels`
            - `Number`, optional
            - number of pixels to which the aperture-mask shall be scaled to
                - i.e. sum over all aperture-mask entries will add up to this value
            - the default is `1`
                - normalization to 1 pixel
        - `outside`
            - `Real`, optional
                - common choices
                    - `NaN`
                    - `0`
                - the default is `0`` 

    Methods
    -------
        - `lp_aperture()`
        - `rect_aperture()`
        - `gauss_aperture()`
        - `lorentz_aperture()`

    Related Structs
    ---------------
        - `ApertureMask`

    Comments
    --------
"""
struct ApertureTemplate
    x_size::Int
    y_size::Int
    x::Number
    y::Number
    normalize::Union{Symbol,Bool}
    npixels::Number
    outside::Real

    #inner constructor (to allow default values)
    function ApertureTemplate(
        x_size::Int, y_size::Int,
        x::Number=0, y::Number=0;
        normalize::Union{Symbol,Bool}=:npixels,
        npixels::Number=1, outside::Real=0,
        )

        @assert normalize in [:max, :npixels, false] "`normalize` has to be one of `:max`, `:npixels`, `false`"

        new(x_size, y_size, x, y, normalize, npixels, outside)
    end
end
"""
    - struct defining an aperture mask

    Fields
    ------
        - `mask`
            - `AbstractMatrix`
            - the generated aperture-mask

    Methods
    -------
        - `plot_aperture!()`
        - `plot_aperture()`

    Related Structs
    ---------------

    Comments
    --------

"""
struct ApertureMask{T <: Number}
    mask::AbstractMatrix{T}
end

#%%aperture shapes
"""
    - function to generate a grid of coordinates used to define the aperture mask

    Parameters
    ----------
        - `apt`
            - `ApertureTemplate`
            - constructor containing hyperparameters of apertures    

    Raises
    ------

    Comments
    --------

    Returns
    -------
        - `coords`
            - `AbstractMatrix`
            - grid of coordinates of size `(apt.x_size,apt.y_size,2)` 
"""
function get_coords(
    apt::ApertureTemplate,
    )::AbstractArray{Real,3}

    coords = cat(
        (1:apt.x_size)  .* ones(apt.y_size)' .- apt.x_size/2,
        (1:apt.y_size)' .* ones(apt.x_size)  .- apt.y_size/2;
        dims=3
    )
    position::AbstractArray{Real,3} = reshape([apt.x,apt.y], 1,1,2)
    position .+= 0.5 #because range starts at `1`!
    coords .-= position

    return coords
end

"""
    - function to execute postprocessing on the generated aperture-mask
    - i.e.
        - normalization
        - setting out-of-aperture values
    
    Parameters
    ----------
        - `apm`
            - `AbstractMatrix`
            - generated aperture mask to be postprocessed
        - `apt`
            - `ApertureTemplate`
            - constructor containing hyperparameters of apertures

    Raises
    ------

    Comments
    --------

    Returns
    -------
        - `apm`
            - `AbstractMatrix`
            - aperture mask after postprocessing
"""
function post_process(
    apm::AbstractMatrix, apt::ApertureTemplate,
    )::AbstractMatrix
    
    if apt.normalize == :max
        apm ./= maximum(apm)
    elseif apt.normalize == :npixels
        apm = isnothing(apt.npixels) ? apm : apm ./ sum(apm) .* apt.npixels
    else
        apm = apm
    end
    apm[(apm .<= 0)] .= apt.outside

    return apm
end

"""
    - method to generate aperturemasks based on the L_p-norms

    Parameters
    ----------
        - `apt`
            - `ApertureTemplate`
            - constructor containing hyperparameters of apertures
        - `radius`
            - `Number`
            - radius of the aperture in a L_p-norm sense
        - `p`
            - `Real`, optional
            - p-parameter in the L_p norm
            - will be passed to `LinearAlgebra.norm()` as `p`
                - `LinearAlgebra.norm(..., p)`
            - the default is `2`
                - L2-norm
                - circular aperture
        - `poly_coeffs`
            - `Vector`, optional
            - polynomial coefficients to describe the radial aperture falloff/gradient
            - will be passed to `Polynomials.Polynomial`
            - the default is `nothing`
                - will be set to `[1]`
                - constant (step-function)
        - `radius_inner`
            - `Number`, optional
            - inner radius of sky ring
            - will create a sky-ring (annulus-like shape) with
                - inner radius = `inner_radius`
                - outer radius = `radius`
            - the default is `0`
                - creates standard aperture

    Raises
    ------

    Returns
    -------
        - `apm`
            - `ApertureMask`
            - generated aperture mask
            - has shape `(apt.size[0],apt.size[1])`

    Comments
    --------
"""
function lp_aperture(
    apt::ApertureTemplate,
    radius::Real, p::Real=2;
    poly_coeffs::Union{Vector,Nothing}=nothing,
    radius_inner::Real=0,
    )::ApertureMask
    
    #default parameters
    poly_coeffs = isnothing(poly_coeffs) ? [1] : poly_coeffs

    #get grid of coordinates
    coords = get_coords(apt)

    #define aperture
    abs_coords = dropdims(mapslices(x -> LinearAlgebra.norm(x, p), coords; dims=3); dims=3)
    apm = Float32.(((radius_inner .<= abs_coords) .& (abs_coords .<= radius)))

    #aperture falloff
    poly = Polynomials.Polynomial(poly_coeffs)
    apm .*= poly.(abs_coords)

    #postprocess aperture
    apm = post_process(apm, apt)

    return ApertureMask(apm)
end

"""
    - method to generate rectangular apertures
    - aperture will be mask of zeros and ones scaled by `apt.npixels`

    Parameters
    ----------
        - `apt`
            - `ApertureTemplate`
            - constructor containing hyperparameters of apertures
        - `width`
            - `Real`
            - width of the aperture
        - `height`
            - `Real`
            - height of the aperture
        - `width_inner`
            - `Real`, optional
            - width of the inner bound of the sky ring
            - will create a sky-ring (donut-like shape) with
                - inner width = `inner_width`
                - outer width = `width`
            - the default is `0`
                - creates standard aperture
        - `height_inner`
            - `Real`, optional
            - height of the inner bound of the sky ring
            - will create a sky-ring (donut-like shape) with
                - inner height = `inner_height`
                - outer height = `height`
            - the default is `0`
                - creates standard aperture

    Raises
    ------

    Returns
    -------
        - `apm`
            - `ApertureMask`
            - generated aperture mask
            - has shape `(apt.size[0],apt.size[1])`

    Comments
    --------
"""
function rect_aperture(
    apt::ApertureTemplate,
    width::Real, height::Real;
    width_inner::Real=0, height_inner::Real=0,
    )::ApertureMask

    #get grid of coordinates
    coords = get_coords(apt)

    #outer aperture
    aperture_outer_x = (-(width/2)  .<= coords[:,:,1]) .& (coords[:,:,1] .<= (width/2)) 
    aperture_outer_y = (-(height/2) .<= coords[:,:,2]) .& (coords[:,:,2] .<= (height/2))
    aperture_outer = aperture_outer_x .& aperture_outer_y
    
    #inner aperture
    aperture_inner_x = (-(width_inner/2)  .>= coords[:,:,1]) .| (coords[:,:,1] .>= (width_inner/2)) 
    aperture_inner_y = (-(height_inner/2) .>= coords[:,:,2]) .| (coords[:,:,2] .>= (height_inner/2))
    aperture_inner = aperture_inner_x .| aperture_inner_y

    #combine
    apm = aperture_inner .& aperture_outer
    
    #postprocess aperture
    apm = post_process(apm, apt)


    return ApertureMask(apm)

end

"""
    - method to generate apertures based on gaussian distributions
    - returns aperture normalized to `apt.npixels`

    Parameters
    ----------
        - `apt`
            - `ApertureTemplate`
            - constructor containing hyperparameters of apertures    
        - `radius`
            - `Real`, optional
            - constraint space for the aperture
            - anything outside `radius` will be set to `apt.outside`
            - the default is `Inf`
                - unconstrained
        - `p`
            - `Real`, optional
            - p-parameter in the L_p norm
            - will be passed to `LinearAlgebra.norm()` as `p`
                - `LinearAlgebra.norm(..., p=p)`
            - utilized in
                - computing the contraint space
                    - i.e. everything outside `radius` w.r.t. the L-p norm will be set to `apt.outside`
                - as norm in the exponent of the gaussian if `lp==true`
                    - will use L-p norm at expense of utilizing a covariance matrix
                    - useful to generate balls with gaussian-like decaying mask-values
            - the default is `2`
                - L2-norm
                - circular aperture
        - `covariance`
            - `Real`, `AbstractMatrix{Real}`, optional
            - covariance matrix in the exponent of the 2d gaussian
            - has to have size `(2,2)`
            - if `Real`
                - will be interpreted as matrix with diagnoal values set to `covariance`
            - otherwise
                - will be interpreted as the covariance matrix
            - the default is `1`
        - `lp`
            - `Bool`, optional
            - whether to use the L-p norm in the exponent of the gaussian instead of the standard expression
            - the defaul is `false`
        - `radius_inner`
            - `Number`, optional
            - inner radius of sky ring
            - will create a sky-ring (donut-like shape) with
                - inner radius = `inner_radius`
                - outer radius = `radius`
            - only applied if `p!=0`
            - the default is `0`
                - creates standard aperture                 

    Raises
    ------
        - `ArgumentError`
            - in case `covariance` is a `AbstractMatrix` and `lp==true`

    Returns
    -------
        - `apm`
            - `ApertureMask`
            - generated aperture mask
            - has shape `(apt.size[0],apt.size[1])`

    Comments
    --------
"""
function gauss_aperture(
    apt::ApertureTemplate,
    radius::Real=Inf,
    p::Real=2;
    covariance::Union{AbstractMatrix{T},Real}=1,
    lp::Bool=false,
    radius_inner::Real=0,
    )::ApertureMask where T <: Real

    #preliminary checks
    if isa(covariance, AbstractMatrix) && lp
        throw(ArgumentError("`covariance` has to be a `Real` number if `lp==true`!"))
    end

    #get grid of coordinates
    coords = get_coords(apt)

    #generate unconstrained aperture
    if lp
        exp2use = dropdims(mapslices(x -> LinearAlgebra.norm(x, p) ./ covariance, coords; dims=3); dims=3)
    else
        exp2use = dropdims(mapslices(x -> x' * LinearAlgebra.inv(covariance) * x, coords; dims=3); dims=3)
    end
    apm = exp.(-exp2use ./ 2)

    #constrain via lp-norm
    if p != 0
        abs_coords = dropdims(mapslices(x -> LinearAlgebra.norm(x, p), coords; dims=3); dims=3)
        lp_mask_outer = abs_coords .<= radius   #outer bound
        lp_mask_inner = radius_inner .<= abs_coords   #inner bound
        lp_mask = lp_mask_inner .& lp_mask_outer
        apm[.~ lp_mask] .= 0
    end

    #postprocess aperture
    apm = post_process(apm, apt)

    return ApertureMask(apm)
end

"""
    - method to generate apertures following a 2d Lorentzian-profile
    - returns aperture normalized to `apt.npixels`

    Parameters
    ----------
        - `apt`
            - `ApertureTemplate`
            - constructor containing hyperparameters of apertures
        - `fwhm_x`
            - `Real`, optional
            - full-width-half-maximum of the lorentzian in x-direction
            - determines spread of the distribution in x
            - the default is `1`
        - `fwhm_y`
            - `Real`, optional
            - full-width-half-maximum of the lorentzian in y-direction
            - determines spread of the distribution in y
            - the default is `1`
        - `radius`
            - `Real`, optional
            - constraint space for the aperture
            - anything outside `radius` will be set to `apt.outside`
            - the default is `Inf`
                - unconstrained
        - `p`
            - `Real`, optional
            - p-parameter in the L_p norm
            - will be passed to `LinearAlgebra.norm()` as `p`
                - `LinearAlgebra.norm(..., p=p)
            - the default is `2`
                - L2-norm
                - circular aperture
        - `radius_inner`
            - `Number`, optional
            - inner radius of sky ring
            - will create a sky-ring (donut-like shape) with
                - inner radius = `inner_radius`
                - outer radius = `radius`
            - only applied if `p!=0`
            - the default is `0`
                - creates standard aperture 

    Raises
    ------

    Returns
    -------
        - `apm`
            - `ApertureMask`
            - generated aperture mask
            - has shape `(apt.size[0],apt.size[1])`

    Comments
    --------
"""
function lorentz_aperture(
    apt::ApertureTemplate,
    fwhm_x::Real=1,
    fwhm_y::Real=1,
    radius::Real=Inf,
    p::Real=0;
    radius_inner::Real=0,
    )::ApertureMask
    
    #get grid of coordinates
    coords = get_coords(apt)

    #compute 2d lorentzian profile
    apm_x = lorentz.(coords[:,:,1]; p0=0, fwhm=fwhm_x)  #`p0` included in `coords`
    apm_y = lorentz.(coords[:,:,2]; p0=0, fwhm=fwhm_y)  #`p0` included in `coords`
    apm = apm_x .* apm_y

    #constrain via lp-norm
    if p != 0
        abs_coords = dropdims(mapslices(x -> LinearAlgebra.norm(x, p), coords; dims=3); dims=3)
        lp_mask_outer = abs_coords .<= radius   #outer bound
        lp_mask_inner = radius_inner .<= abs_coords   #inner bound
        lp_mask = lp_mask_inner .& lp_mask_outer
        apm[.~ lp_mask] .= 0
    end

    return ApertureMask(apm)

end

#plotting apertures
"""
    - function to plot an aperture as as a combination of rectangles
    
    Parameters
    ----------
        - `plt`
            - `Plots.Plot`
            - panel to plot into
        - `ap_mask`
            - `ApertureShapes.ApertureMask`
            - the aperture-mask to plot
        - `xpix`
            - `AbstractArray`, `Nothing`, optional
            - x-coordinates of the pixels to plot
            - the default is `nothing`
                - will use the `xlims` of `plt` to generate coordinates
                    - `xpix = range(1,Plots.xlims(plt)[2])`
                - thus, will plot aperture position in relation to `x=1`
        - `ypix`
            - `AbstractArray`, `Nothing`, optional
            - y-coordinates of the pixels to plot
            - the default is `nothing`
                - will use the `ylims` of `plt` to generate coordinates
                    - `ypix = range(1,Plots.ylims(plt)[2])`
                - thus, will plot aperture position in relation to `y=1`
        - `linecolor`
            - `Any`, optional
            - any type that is accepted for the `linecolor` kwarg in `Plots.plot`
            - color of the outlines of the aperture pixels
            - the default is `1`
                - first color in the loaded color palette
        - `linealpha`
            - `Symbol`, `Real`, optional
            - opacity values to assign to the outlines of the aperture pixels
            - if `Symbol` has to be one of
                - `:ap_mask`
                    - `linealpha` will be set in accordance with `ap_mask`
            - if `Real`
                - will be set that the specified constant value across the grid
            - the default is `:ap_mask`
        - `fillcolor`
            - `Any`, optional
            - any type that is accepted for the `linecolor` kwarg in `Plots.plot`
            - color to use for filling in the aperture pixels
            - the default is `:line`
                - will use the color assigned to the line (`linecolor`)
        - `fillalpha`
            - `Symbol`, `Real`, optional
            - opacity values to assign to the filling of the aperture pixels
            - if `Symbol` has to be one of
                - `:ap_mask`
                    - `fillalpha` will be set in accordance with `ap_mask`
                - `:line`
                    - `fillalpha` will be set to the same value as `linealpha`
            - if `Real`
                - will be set that the specified constant value across the grid
            - the default is `0`
                - no filling
        - `plot_kwargs`
            - `Vararg`
            - kwargs to pass to `plot`
            - note that this will be passed to every cell in the aperture!

    Raises
    ------

    Returns
    -------
        - `plt`
            - `Plots.Plot`
            - created panel    

    Comments
    --------
        - it is suggested to use `ap_mask` where the out-of aperture values are `NaN` if one sets `linealpha` and `fillalpha` to `Real` values
"""
function plot_aperture!(plt::Plots.Plot,
    ap_mask::ApertureMask,
    xpix::Union{AbstractArray,Nothing}=nothing, ypix::Union{AbstractArray,Nothing}=nothing;
    linecolor::Any=1, linealpha::Union{Symbol,Real}=:ap_mask,
    fillcolor::Any=:line, fillalpha::Union{Symbol,Real}=0.,
    plot_kwargs...
    )

    @assert isa(linealpha, Real) | (linealpha in [:ap_mask])        "`linealpha` has to be either a `Real` number or one of `:ap_mask`"
    @assert isa(fillalpha, Real) | (fillalpha in [:ap_mask, :line]) "`fillalpha` has to be either a `Real` number or one of `:ap_mask`, `:line`"

    #default parameters
    if linealpha == :ap_mask
        linealpha = ap_mask.mask ./ nanmaximum(ap_mask.mask)
    else
        linealpha = ones(size(ap_mask.mask)) .* linealpha
    end
    fillcolor = fillcolor == :line ? linecolor : fillcolor
    if fillalpha == :line
        fillalpha = linealpha
    elseif fillalpha == :ap_mask
        fillalpha = ap_mask.mask ./ nanmaximum(ap_mask.mask)
    else
        fillalpha = ones(size(ap_mask.mask)) .* fillalpha
    end

    #get coordinate range to plot to
    xpix = isnothing(xpix) ? Float32.(collect(axes(ap_mask.mask, 1))) : Float32.(xpix)
    ypix = isnothing(ypix) ? Float32.(collect(axes(ap_mask.mask, 2))) : Float32.(ypix)

    #offset to align with `heatmap()`
    xpix .= xpix .- 0.5
    ypix .= ypix .- 0.5

    #plot rectangles (aperture pixels)
    for (i, x) in enumerate(xpix)
        for (j, y) in enumerate(ypix)
            x2plot = [x, x+1, x+1, x]    #coords of the cell
            y2plot = [y, y, y+1, y+1]    #coords of the cell

            #plot cell #ignore `NaN` in plotting
            if ~isnan(ap_mask.mask[i,j])
                # plot!(plt, y2plot, x2plot; linecolor=:red, linealpha=ap_mask.mask[i,j]/nanmaximum(ap_mask.mask), label="", plot_kwargs...)
                plot!(plt,
                    Plots.Shape(y2plot, x2plot);
                    linecolor=linecolor, linealpha=linealpha[i,j],
                    fillcolor=fillcolor, fillalpha=fillalpha[i,j],
                    label="",
                    plot_kwargs...
                )
            end
        end
    end

end
function plot_aperture(
    ap_mask::ApertureMask,
    xpix::Union{AbstractArray,Nothing}=nothing, ypix::Union{AbstractArray,Nothing}=nothing;
    linecolor::Any=1, linealpha::Union{Symbol,Real}=:ap_mask,
    fillcolor::Any=:line, fillalpha::Union{Symbol,Real}=0,
    plot_kwargs...
    )::Plots.Plot

    #plot
    plt = plot()
    plot_aperture!(plt, ap_mask, xpix, ypix;
        linecolor=linecolor, linealpha=linealpha,
        fillcolor=fillcolor, fillalpha=fillalpha,
        plot_kwargs...
    )

    return plt
end

#visualizing
"""
    - function to pretty-print a generated aperture

    Parameters
    ----------
        - `apm`
            - `ApertureMask`
            - the aperture mask to display

    Raises
    ------

    Returns
    -------

    Comments
    --------
        - suggested only for reasonably small apertures

"""
function print_aperture(apm::ApertureMask)
    for i in axes(apm.mask, 2)
        for j in axes(apm.mask, 1)
            if apm.mask[j,i] > 1e-12
                print(@sprintf("%5.3f,", apm.mask[j,i]))
            else
                print("     ,")
            end
        end
        println()
    end
end


end #module

