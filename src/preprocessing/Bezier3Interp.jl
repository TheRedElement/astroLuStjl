
"""
    - module implementing interpolation via cubic bezier curves
    - a detailed explanation for the derivations can be found here
        - [https://omaraflak.medium.com/b%C3%A9zier-interpolation-8033e9a262c2](https://omaraflak.medium.com/b%C3%A9zier-interpolation-8033e9a262c2) (last access: 2024/10/07)

    Structs
    -------
        - `Bz3Interp`

    Functions
    ---------
        - `bezier()`
        - `fit()`
        - `transform()`

    Extended Functions
    ------------------
        - `Plots.plot()`
        - `Plots.plot!()`

    Dependencies
    ------------
        - `LaTeXStrings`
        - `LinearAlgebra`
        - `Plots`

    Examples
    --------
        - see [../src_demos/Bezier3Interp_demo.jl](../src_demos/Bezier3Interp_demo.jl)
"""

module Bezier3Interp

#%%imports
using LaTeXStrings
using LinearAlgebra
using Plots

#import for extending
import Plots: plot, plot!

#intradependencies
include(joinpath(@__DIR__, "./FormattingUtils.jl"))

#%%exports
export Bz3Interp
export bezier
export fit
export transform
export plot
export plot!

#%%definitions

#######################################
#helper functions
"""
    - function defining a bezier curve of arbitrary degree
    
    Parameters
    ----------
        - `x`
            - `Real`
            - point at which to evaluate the curve
        - `k`
            - `Vector{Real}`
            - control points of the bezier curve
            - the curve will have a degree of `size(k,1)-1`

    Raises
    ------

    Returns
    -------
        `y`
            - `Real`
            - the defined bezier curve evaluated at `x`

    Comments
    --------
"""
function bezier(
    x::Real;
    k::Vector{T}
    )::Real where {T <: Real}
    deg = size(k, 1) - 1
    terms = [binomial(deg, i) * x^i * (1-x)^(deg-i) * k[i+1] for i in (eachindex(k) .- 1)]
    y = sum(terms)
    return y
end


#######################################
#%%cubic bezier interpolation
"""
    - struct defining the cubic bezier interpolation

    Fields
    ------
        - `K`
            - `AbstractArray{Real,3}`, optional
            - bezier control points of each segment
                - a segment is the trajectory between two neighboring points that shall be interpolated
            - the default is `Array{Real}(undef,0,0,0)`
                - value if transformer is not fitted
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
struct Bz3Interp{T <: Real}
    K::AbstractArray{T,3}
    state::Symbol

    #inner constructor (to allow default values)
    function Bz3Interp(
        K::AbstractArray{T,3}=Array{Real}(undef,0,0,0);
        state::Symbol=:init,
        ) where {T <: Real}

        @assert state in [:init,:fitted] ArgumentError("`state` has to be one of `:init`, `:fitted` but is `$state`!")

        new{T}(K, state)
    end

end

"""
    - function to fit `Bz3Interp` to data

    Parameters
    ----------
        - `b3i`
            - `Bz3Interp`
            - struct containing hyperparameters of the model
        - `x`
            - `AbstractMatrix{T <: Real}`
            - data to be interpolated
            - has to have size `(ndims,nsamples)`
            - can be of arbitrary dimensionality (ndims arbitrary)
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`
    
    Raises
    ------

    Returns
    -------
        - `b3i`
            - `Bz3Interp`
            - fitted version of `b3i`
                - `state` set to `:fitted`

    Comments
    --------
"""
function fit(
    b3i::Bz3Interp,
    x::AbstractMatrix{T};
    verbose::Int=0,
    )::Bz3Interp where {T <: Real}

    n = size(x, 2) - 1

    #coeff matrix
    c = 4 .* Matrix(I, n, n)
    c[diagind(c, 1)] .= 1
    c[diagind(c, -1)] .= 1
    c[1,1] = 2
    c[n,n] = 7
    c[n,n-1] = 2

    #bezier coeff vector
    p       = 2 .* (2 .* x[:,1:end-1] .+ x[:,2:end])
    p[:,1]    .= x[:,1] .+ 2*x[:,2]
    p[:,end]  .= 8*x[:,end-1] .+ x[:,end]

    #solve system
    a = (c \ p')'
    b = zeros(size(x,1),n)
    b[:,1:n-1] = 2 .* x[:,2:end-1] .- a[:,2:end]
    b[:,n] = (a[:,end] .+ x[:,end]) / 2     #size(x,2) = size(a,2) + 1

    #get matrix of coeffs
    K = cat(x[:,1:end-1], a, b, x[:,2:end]; dims=3) #(ncoords,nsegments,npoints)

    return Bz3Interp(K; state=:fitted)
end

"""
    - function to transform data based on parameters stored in `Bz3Interp`

    Parameters
    ----------
        - `b3i`
            - `Bz3Interp`
            - fitted version of `b3i`
                - `state` set to `:fitted`
        - `n`
            - `Int`
            - number of datapoints the interpolated series shall have
            - will be distributed across the number of segments
                - i.e. one less than the number of datapoints used for fitting
            - to ensure your number of points is met, make sure to pass a number divisible by the number of segments

    Raises
    ------
        - `AssertionError`
            - if the model has not been fit yet

    Returns
    -------
        - `path`
            - `AbstractArray`
            - has size `(ndims,n)`
                - if `n` is divisible by the number of segments
                - `ndims` is equal to the number of coordinates of the input data

    Comments
    --------
"""
function transform(
    b3i::Bz3Interp,
    n::Int,
    )::AbstractArray

    @assert b3i.state == :fitted "`b3i` has not been fitted yet. make sure to call `fit(b3i,...)` before transforming"

    ndims, nsegments, npoints = size(b3i.K)

    x_interp = range(0,1;length=div(n,nsegments))

    curves = mapslices(x -> bezier.(x_interp; k=x), b3i.K; dims=3)
    path = reshape(permutedims(curves, (1,3,2)), ndims, :)

    return path
end

"""
    - extensions to `Plots.plot!()` and `Plots.plot()`
    - plots 2d projection of the result obtained using `Bz3Interp`

    Parameters
    ----------
        - `plt`
            - `Plots.Plot`
            - panel to plot into
        - `b3i`
            - `Bz3Interp`
            - fitted instance of `Bz3Interp`
            - first two coordinates of the result will be plotted
        - `x`
            - `Matrix`, optional
            - input datapoints that got interpolated
            - first two coordinates
            - the default is `nothing`
                - ignored


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
    b3i::Bz3Interp;
    x::Union{Matrix,Nothing}=nothing
    )

    #get interpolation result
    x_interp = transform(b3i, size(b3i.K,2)*20)

    #input scatter (only if `x` and ` y` were passed)
    if ~isnothing(x)
        scatter!(plt, x[1,:], x[2,:]; label="Input")
    end
    
    #representative curve
    plot!(plt,  #outline
        x_interp[1,:], x_interp[2,:];
        label="Bezier Interpolation"
    )

end
function Plots.plot(
    b3i::Bz3Interp;
    x::Union{Matrix,Nothing}=nothing
    )::Plots.Plot
    plt = plot(;xlabel=L"$x$", ylabel=L"$y$", title="Cubic Bezier Interpolation")
    plot!(plt, b3i; x=x,)
    return plt
end

end #module