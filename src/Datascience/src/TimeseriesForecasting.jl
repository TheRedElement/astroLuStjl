
"""
    - module implementing methods for timeseries forecasting

    Constants
    ---------

    Structs
    -------
        - `ARIMA`
        - `AutoRegressor`
        - `MovingAverage`
        - `RecursivePolynomialRegressor`

    Functions
    ---------
        - `difference()`
        - `un_difference()`        
        - `fit()`
        - `predict()`
        - `plot()`

    Extended Functions
    ------------------

    Dependencies
    ------------
        - `Plots`
        - `NaNStatistics`

    Comments
    --------

    Examples
    --------
        - see [TimeseriesForecasting_demo.jl](../../demos/Datascience/TimeseriesForecasting_demo.jl)
"""

module TimeseriesForecasting

#%%imports
using NaNStatistics
using Plots

#import for extending
import Plots: plot, plot!

#intradependencies
include("../../Styles/Styles.jl")
include("../../Preprocessing/Preprocessing.jl")
using .Styles.FormattingUtils
using .Preprocessing.Subsampling

#%%exports
export ARIMA
export AutoRegressor
export MovingAverage
export RecursivePolynomialRegressor
export difference
export un_difference
export ar_process
export fit
export predict
export plot, plot!

#%%definitions

#######################################
#helper functions
"""
    - function to generate data following some autoregressive process

    Parameters
    ----------
        - `ntimes`
            - `Int`
            - number of timesteps to generate
        - `h`
            - `Vector`
            - filter coefficients of the process
            - step t will be generated according to \$x_t = \\sum_{k=1}^K x_{t-k} h_k\$
                - K is hereby the  order of the process (nubmer of filter coefficients)
        - `offset`
            - `Real`, optional
            - some offset to the series
            - shifts the series mean
            - the default is `0.0`
        - `var`
            - `Real`, optional
            - variance of the process
            - the default is `0.0`
                - no variance
                - horizontal line
    
    Raises
    ------

    Returns
    -------
        - `z`
            - `Vector`
            - dataseries generated following the autoregressive process

    Comments
    --------
"""
function ar_process(
    ntimes::Int,
    h::Vector;
    offset::Real=0.0, var::Real=0.0,
    )::Vector

    #number of coeffs
    ncoeffs = size(h,1)

    z = randn(ntimes+ncoeffs).*sqrt(var) .+ offset #baseline noise + offset
    for t in ncoeffs.+(1:ntimes)
        for i in 1:ncoeffs
            z[t] += h[i]*z[t-i]
        end
    end
    z = z[(ncoeffs+1):end]

    return z
end

"""
    - function to apply differencing on some autoregressive series `x`
    - will remove trends of degree `d`

    Parameters
    ----------
        - `x`
            - `Vector`
            - series to which apply differencing on
        - `d`
            - `Int`
            - differencing degree
            - autoregressive trends up to degree `d` will be removed from the series

    Raises
    ------

    Returns
    -------
        - `x_diff`
            - `Vector`
            - `x` after differencing
            - trends up to degree `d` are removed

    Comments
    --------
"""
function difference(
    x::Vector,
    d::Int,
    )::Vector

    x_diff = copy(x)
    for _ in 1:d
        x_diff = diff(x_diff)
    end
    return x_diff
end
"""
    - function to to restore original from some autoregressive series that got `difference` applied to it

    Parameters
    ----------
        - `x`
            - `Vector`
            - series to that shall be reverted to its original form
            - autoregressive trends of degree `d` will be added
        - `d`
            - `Int`
            - differencing degree
            - autoregressive trends up to degree `d` will be added back to the series


    Raises
    ------

    Returns
    -------
        - `x_undiff`
            - `Vector`
            - `x` after un_differencing
            - trends up to degree `d` added back to `x`

    Comments
    --------
"""
function un_difference(
    x::Vector,
    d::Int,
    )::Vector

    x_undiff = copy(x)
    for _ in 1:d
        x_undiff = cumsum(x_undiff)
    end
    return x_undiff
end

#######################################
#Auto-Regression
"""
    - struct defining an Auto-Regressive estimator

    Fields
    ------
        - `p`
            - `Int`
            - order of the `AutoRegressor`
        - `phi`
            - `Vector`, optional
            - parameters of the fitted model
            - the default is `nothing`
        - `residuals`
            - `Vector`, optional
            - residuals of the fitted model
            - the default is `nothing`
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
        - `predict()`
        - `plot()`
        - `plot!()`

    Comments
    --------
"""
struct AutoRegressor
    p::Int
    phi::Union{Vector,Nothing}
    residuals::Union{Vector,Nothing}
    state::Symbol
    
    #inner constructor (to allow default values)
    function AutoRegressor(
        p::Int=1;
        phi::Union{Vector,Nothing}=nothing,
        residuals::Union{Vector,Nothing}=nothing,
        state::Symbol=:init,
        )

        @assert state in [:init,:fitted] ArgumentError("`state` has to be one of `:init`, `:fitted` but is `$state`!")

        new(p, phi, residuals, state)    
    end
end

"""
    - method to fit an `AutoRegressor` estimator

    Parameters
    ----------
        - `ar`
            - `AutoRegressor`
            - instance of `AutoRegressor` containing the model initialization
        - `x`
            - `Vector`
            - some autoregressive series to base the estimation on
            - has to
                - be stationary (constant mean)
                - not contain seasonal trends
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`

    Raises
    ------

    Returns
    -------
        - `ar`
            - `AutoRegressor`
            - fitted version of `ar`

    Comments
    --------
"""
function fit(
    ar::AutoRegressor,
    x::Vector;
    verbose::Int=0,
    )::AutoRegressor

    xy_mat = Subsampling.get_subsequences(x, ar.p+1; dim=1, stride=1, verbose=0)
    x_ar = xy_mat[1:end-1,:]'
    x_ar = hcat(ones(size(x_ar,1), 1), x_ar) #add ones for intercept
    y_ar = xy_mat[end,:]

    #fit via least squares
    phi = x_ar \ y_ar

    #get residuals of prediction
    residuals = -(x_ar * phi .- y_ar)

    return AutoRegressor(ar.p; phi=phi, residuals=residuals, state=:fitted)
end

"""
    - function to predict using a fitted instance of `AutoRegressor`
    - will forecast `n2pred` steps in the future

    Parameters
    ----------
        - `ar`
            - `AutoRegressor`
            - fitted instance of `AutoRegressor`
        - `x`
            - `Vector`
            - autoregressive series to base the prediction on
        - `n2pred`
            - `Int`
            - forcast window
            - will forcast `n2pred` new steps
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`
    
    Raises
    ------
        - `AssertionError`
            - if the model has not been fit yet

    Returns
    -------
        - `x_pred`
            - `Vector`
            - `n2pred` next steps based on the series `x` using `ar`

    Comments
    --------

"""
function predict(
    ar::AutoRegressor,
    x::Vector,
    n2pred::Int;
    verbose::Int=0,
    )::Vector
    
    @assert ar.state == :fitted "`ar` has not been fitted yet. make sure to call `fit(ar,...)` before predicting!"
    
    #get order and parameters
    p = size(ar.phi,1) - 1
    phi = copy(ar.phi)
    
    #separate parameters
    intercept = splice!(phi,1)  #single out intercept (otherwise will get modified when updating inputs)
    
    #init in- and output
    x_pred = zeros(n2pred)
    x_in = x[end-p+1:end]
    for n in 1:n2pred

        FormattingUtils.printlog(
            "Iteration $n\n";
            context="predict(AutoRegressor)",
            type=:INFO,
            verbose=verbose,
        )        
                
        #make prediction
        x_pred[n] += (intercept + (x_in' * phi))

        #update input
        x_in = vcat(x_in, x_pred[n:n])[2:end]
    end

    return x_pred    
end

"""
    - extensions to `Plots.plot!()` and `Plots.plot()`

    Parameters
    ----------
        - `plt`
            - `Plots.Plot`
            - panel to plot into    
        - `ar`
            - `AutoRegressor`
            - model to visualize
        - `x`
            - `Vector`
            - input vector to use for visualizing `ar`
    
    Raises
    ------
        - `AssertionError`
            - if the model has not been fit yet

    Returns
    -------
        - `plt`
            - `Plots.Plot`
            - created panel
    
    Comments
    --------
    
"""
function Plots.plot!(
    plt::Plots.Plot,
    ar::AutoRegressor,
    x::Vector,
    )

    @assert ar.state == :fitted "`ar` has not been fitted yet. make sure to call `fit(ar,...)` before plotting"

    xy_mat = Subsampling.get_subsequences(x, ar.p+1; dim=1, stride=1, verbose=0)
    x_ar = xy_mat[1:end-1,:]'
    x_ar = hcat(ones(size(x_ar,1), 1), x_ar) #add ones for intercept
    y_ar = xy_mat[end,:]

    y_pred = x_ar * ar.phi

    plot!(plt, axes(y_pred,1), y_pred;  label="AR Prediction")
    plot!(plt, axes(y_ar,1), y_ar;      label="Ground Truth")

end
function Plots.plot(
    ar::AutoRegressor,
    x::Vector,
    )::Plots.Plot

    plt = plot()
    plot!(plt, ar, x)

    return plt

end

#######################################
#Moving Average
"""
    - struct defining a Moving-Average estimator

    Fields
    ------
        - `q`
            - `Int`
            - order of the `MovingAverage`
        - `theta`
            - `Vector`, optional
            - parameters of the fitted model
            - the default is `nothing`
        - `residuals`
            - `Vector`, optional
            - residuals of the fitted model
            - the default is `nothing`
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
        - `predict()`
        - `plot()`
        - `plot!()`

    Comments
    --------
"""
struct MovingAverage
    q::Int
    theta::Union{Vector,Nothing}
    state::Symbol
    
    #inner constructor (to allow default values)
    function MovingAverage(
        q::Int=1;
        theta::Union{Vector,Nothing}=nothing,
        state::Symbol=:init,
        )

        @assert state in [:init,:fitted] ArgumentError("`state` has to be one of `:init`, `:fitted` but is `$state`!")

        new(q, theta, state)
    end

end

"""
    - method to fit a `MovingAverage` estimator

    Parameters
    ----------
        - `ma`
            - `MovingAverage`
            - instance of `MovingAverage` containing the model initialization
        - `residuals`
            - `Vector`
            - residuals between some autoregressive series and predictions of it
            - defined as `ground_truth - prediction`
            - estimate based on `residuals`
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`

    Raises
    ------

    Returns
    -------
        - `ma`
            - `MovingAverage`
            - fitted version of `ma`

    Comments
    --------
"""
function fit(
    ma::MovingAverage,
    residuals::Vector;
    verbose::Int=0,
    )::MovingAverage

    xy_mat = Subsampling.get_subsequences(residuals, ma.q+1; dim=1, stride=1, verbose=0)
    x_ma = xy_mat[1:end-1,:]'
    x_ma = hcat(ones(size(x_ma,1), 1), x_ma) #add ones for intercept
    y_ma = xy_mat[end,:]

    #fit via least squares
    theta = x_ma \ y_ma

    return MovingAverage(ma.q; theta=theta, state=:fitted)
end

"""
    - function to predict using a fitted instance of `MovingAverage`
    - will forecast `n2pred` steps in the future

    Parameters
    ----------
        - `arima`
            - `ARIMA`
            - fitted instance of `ARIMA`
        - `residuals`
            - `Vector`
            - residuals between some autoregressive series and predictions of it
            - defined as `ground_truth - prediction`
            - prediction based on `residuals`
        - `n2pred`
            - `Int`
            - forcast window
            - will forcast `n2pred` new steps
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`
    
    Raises
    ------
        - `AssertionError`
            - if the model has not been fit yet

    Returns
    -------
        - `residuals_pred`
            - `Vector`
            - `n2pred` next steps based on the series `residuals` using `ma`

    Comments
    --------

"""
function predict(
    ma::MovingAverage,
    residuals::Vector,
    n2pred::Int;
    verbose::Int=0,
    )::Vector

    @assert ma.state == :fitted "`ma` has not been fitted yet. make sure to call `fit(ma,...)` before predicting!"

    #get order and parameters
    q = size(ma.theta,1) - 1
    theta = copy(ma.theta)

    #separate parameters
    intercept = splice!(theta,1)  #single out intercept (otherwise will get modified when updating inputs)


    residuals_pred = zeros(n2pred)
    residuals_in = residuals[end-q+1:end]
    for n in 1:n2pred

        FormattingUtils.printlog(
            "Iteration $n\n";
            context="predict(MovingAverage)",
            type=:INFO,
            verbose=verbose,
        )      

        #make prediction
        residuals_pred[n] += (intercept + (residuals_in' * theta))

        #update input
        residuals_in = vcat(residuals_in, residuals_pred[n:n])[2:end]
    end

    return residuals_pred

end

"""
    - extensions to `Plots.plot!()` and `Plots.plot()`

    Parameters
    ----------
        - `plt`
            - `Plots.Plot`
            - panel to plot into    
        - `ma`
            - `MovingAverage`
            - model to visualize
        - `residuals`
            - `Vector`
            - input residuals to used for fitting `ma`
            - defined as `ground_truth - prediction`
        - `x`
            - `Vector`, optional
            - ground truth `residuals` stem from
            - the defaul tis `nothing`
                - ignored

    Raises
    ------
        - `AssertionError`
            - if the model has not been fit yet

    Returns
    -------
        - `plt`
            - `Plots.Plot`
            - created panel
    
    Comments
    --------
    
"""
function Plots.plot!(
    plt::Plots.Plot,
    ma::MovingAverage,
    residuals::Vector,
    x::Union{Vector,Nothing}=nothing,
    )

    @assert ma.state == :fitted "`ma` has not been fitted yet. make sure to call `fit(ma,...)` before plotting"

    xy_resids_mat = Subsampling.get_subsequences(residuals, ma.q+1; dim=1, stride=1, verbose=0)
    x_resids_ma = xy_resids_mat[1:end-1,:]'
    x_resids_ma = hcat(ones(size(x_resids_ma,1), 1), x_resids_ma) #add ones for intercept
    y_resids_ma = xy_resids_mat[end,:]

    y_pred = x_resids_ma * ma.theta
    if ~isnothing(x)
        t_x = axes(x,1)
        t_ma = t_x[end-size(y_pred, 1)+1:end]
        x = x
    else
        t_x = [NaN]
        t_ma = axes(y_pred)
        x = [NaN]
    end

    plot!(plt, t_ma, y_pred;        label="MA Prediction")
    plot!(plt, t_x, x;              label="Ground Truth")
    plot!(plt, t_ma, y_resids_ma;   label="Residuals", ls=:dashdot)

end
function Plots.plot(
    ma::MovingAverage,
    residuals::Vector,
    x::Union{Vector,Nothing}=nothing,
    )::Plots.Plot

    plt = plot()
    plot!(plt, ma, residuals, x)

    return plt

end

#######################################
#ARIMA
"""
    - struct defining an ARIMA (Auto Regressive Integrated Moving Average) estimator

    Fields
    ------
        - `p`
            - `Int`
            - order of the `AutoRegressor`
        - `d`
            - `Int`
            - order of the differencing
        - `q`
            - `Int`
            - order of the `MovingAverage`
        - `ar`
            - `AutoRegressor`, optional
            - model used in the AR part
            - will be build during `fit()`
            - the default is `nothing`
        - `ma`
            - `MovingAverage`, optional
            - model used in the MA part
            - will be build during `fit()`
            - the default is `nothing`
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
        - `predict()`
        - `plot()`
        - `plot!()`

    Comments
    --------
"""
struct ARIMA
    p::Int
    d::Int
    q::Int
    ar::Union{AutoRegressor,Nothing}
    ma::Union{MovingAverage,Nothing}
    state::Symbol

    #inner constructor (to allow default values)
    function ARIMA(
        p::Int,
        d::Int,
        q::Int;
        ar::Union{AutoRegressor,Nothing}=nothing,
        ma::Union{MovingAverage,Nothing}=nothing,
        state::Symbol=:init,
        )

        @assert state in [:init,:fitted] ArgumentError("`state` has to be one of `:init`, `:fitted` but is `$state`!")

        new(p, d, q, ar, ma, state)
    end
end
"""
    - method to fit an `ARIMA` estimator

    Parameters
    ----------
        - `arima`
            - `ARIMA`
            - instance of `ARIMA` containing the model initialization
        - `x`
            - `Vector`
            - some autoregressive series to base the estimation on
            - shall not contain seasonal trends
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`

    Raises
    ------

    Returns
    -------
        - `arima`
            - `ARIMA`
            - fitted version of `arima`

    Comments
    --------
"""
function fit(
    arima::ARIMA,
    x::Vector;
    verbose::Int=0
    )::ARIMA

    #init submodels
    ar = AutoRegressor(arima.p)
    ma = MovingAverage(arima.q)

    #differencing step
    x_diff = difference(x, arima.d)

    #autoregression step
    ar = fit(ar, x_diff; verbose=verbose)

    #moving average step
    ma = fit(ma, ar.residuals; verbose=verbose)

    # return ARIMA(arima.p, arima.d, arima.q; phi=phi, theta=theta, residuals=last_resids, state=:fitted)
    return ARIMA(arima.p, arima.d, arima.q; ar=ar, ma=ma, state=:fitted)
end

"""
    - function to predict using a fitted instance of `ARIMA`
    - will forecast `n2pred` steps in the future

    Parameters
    ----------
        - `arima`
            - `ARIMA`
            - fitted instance of `ARIMA`
        - `x`
            - `Vector`
            - autoregressive series to base the prediction on
        - `n2pred`
            - `Int`
            - forcast window
            - will forcast `n2pred` new steps
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`
    
    Raises
    ------
        - `AssertionError`
            - if the model has not been fit yet

    Returns
    -------
        - `x_pred`
            - `Vector`
            - `n2pred` next steps based on the series `x` using `arima`

    Comments
    --------

"""
function predict(
    arima::ARIMA,
    x::Vector,
    n2pred::Int;
    verbose::Int=0,
    )::Vector

    @assert arima.state == :fitted "`arima` has not been fitted yet. make sure to call `fit(arima,...)` before predicting!"

    #init in- and output
    x_pred = zeros(n2pred)
    x_in = difference(copy(x), arima.d)                         #apply differencing
    residuals_in = copy(arima.ar.residuals[end-arima.q+1:end])  #get last `q` residuals (to predict with MovingAverage)
    for n in 1:n2pred
        
        FormattingUtils.printlog(
            "Iteration $n\n";
            context="predict(ARIMA)",
            type=:INFO,
            verbose=verbose,
        )        
        
        #prediction with autoregressor (only one step because combination with MA is final prediction)
        x_pred_ar = predict(arima.ar, x_in, 1)[1]
        
        #prediction with moving average (only one step because combination with AR is final prediction)
        x_pred_ma = predict(arima.ma, residuals_in, 1)[1]

        #final prediction (ARMA)
        x_pred[n] = x_pred_ar + x_pred_ma

        #update inputs
        x_in = vcat(x_in, x_pred[n:n])[2:end]
        residuals_in = vcat(residuals_in, [0])[2:end]   #placeholder => future residuals are unknown
    end

    #remove differencing
    x_pred = un_difference([x[end], x_pred...], arima.d)[2:end] #drop first entry because that is still part of the training data

    return x_pred

end

"""
    - extensions to `Plots.plot!()` and `Plots.plot()`

    Parameters
    ----------
        - `plt`
            - `Plots.Plot`
            - panel to plot into    
        - `arima`
            - `ARIMA`
            - model to visualize
        - `x`
            - `Vector`
            - input vector to use for visualizing `arima`
        - `n2pred`
            - `Int`, optional
            - number of steps in the future to predict
            - the default is `5`
    
    Raises
    ------
        - `AssertionError`
            - if the model has not been fit yet

    Returns
    -------
        - `plt`
            - `Plots.Plot`
            - created panel
    
    Comments
    --------
    
"""
function Plots.plot!(
    plt::Plots.Plot,
    arima::ARIMA,
    x::Vector;
    n2pred::Int=5,
    )

    @assert arima.state == :fitted "`arima` has not been fitted yet. make sure to call `fit(arima,...)` before plotting"

    x_pred = predict(arima, x, n2pred; verbose=0)

    plot!(plt, size(x,1).+(1:n2pred), x_pred;  label="ARIMA Forecast")
    plot!(plt, axes(x,1), x;  label="Input")
    # plot!(plt, axes(y_ma,1), y_ma;      label="Ground Truth")

end
function Plots.plot(
    arima::ARIMA,
    x::Vector;
    n2pred::Int=5,
    )::Plots.Plot

    plt = plot()
    plot!(plt, arima, x; n2pred=n2pred)

    return plt

end

#######################################
#Recursive-Polynomial-Regressor
"""
    - struct defining a recursive polynomial regressor based
    - will fit a polynomial for every step and use the polynomial to predict the next step

    Fields
    ------
        - `poly_degree`
            - ``Int`, optional
            - polynomial degree to be used for the extrapolation
            - the default is `2`
        - `dx`
            - `Real`, optional
            - stepsize to take for predicting a new step
            - for every one of `n2pred` steps, will predict at `x[end]+dx`
            - the default is `nothing`
                - will be infered from the input data
                - `nanmedian(diff(x))` will be used            
        - `state`
            - `Symbol`, optional
            - can be one of
            - `:init`
                - the model has been initialized
            - `:fitted`
                - the model has been fitted
            - the default is `:fitted`
                - no need to fit the predictor

    Methods
    -------
        - `fit()`
        - `predict()`
        - `plot()`

    Comments
    --------
"""
struct RecursivePolynomialRegressor
    poly_degree::Int
    dx::Union{Real,Nothing}
    state::Symbol

    #inner constructor (to allow default values)
    function RecursivePolynomialRegressor(
        poly_degree::Int=2;
        dx::Union{Real,Nothing}=nothing,
        state::Symbol=:fitted
        )

        @assert state in [:init,:fitted] ArgumentError("`state` has to be one of `:init`, `:fitted` but is `$state`!")

        new(poly_degree, dx, state)
    end

end

"""
    - function to fit the `RecursivePolynomialRegressor`
    - extrapolates a series `y(x)` with `n2pred` new steps
    - uses initial `y` and `x` as starting point
        - fits polynomial of degree `rpr.deg` to the data
        - computes the fitted polynomial at the next step (`x[end]+dx`)
        - autoregressively repeats until `n2pred` new points are predicted
    
    Parameters
    ----------
        - `rpr`
            - `RecursivePolynomialRegressor`
            - struct containing hyperparameters of the model
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`

    Raises
    ------

    Returns
    -------
        - `rpr`
            - `RecursivePolynomialRegressor`
            - fitted version of `rpr`
                - `state` set to `:fitted`

    Comments
    --------

"""
function fit(
    rpr::RecursivePolynomialRegressor;
    verbose::Int=0,
    )::RecursivePolynomialRegressor
    
    FormattingUtils.printlog(
        "you don't need to call `fit(RecursivePolynomialRegressor, ...)` before `predict(RecursivePolynomialRegressor,...)`...\n";
        context="fit(RecursivePolynomialRegressor)",
        type=:INFO,
        verbose=verbose,
    )

    return RecursivePolynomialRegressor(rpr.poly_degree;
        dx=rpr.dx,
        state=:fitted
    )

end
function fit(    #signature variation
    rpr::RecursivePolynomialRegressor,
    x::Vector, y::Vector;
    verbose::Int=0,
    )::RecursivePolynomialRegressor
        
    return fit(rpr; verbose=verbose)
end

"""
    - function to predict using the fitted predictor
    
    Parameters
    ----------
        - `rpr`
            - `RecursivePolynomialRegressor`
            - fitted version of `RecursivePolynomialRegressor`
        - `y`
            - `Vector`
            - y-values of the data to be extrapolated
        - `n2pred`
            - `Int`
            - number of steps to extrapolate
            - will result in an output covering `n2pred*dx`
        - `x`
            - `Vector`, optional
            - x-values of the data to be extrapolated
            - make sure to choose `x` to have a reasonable range
                - otherwise the exponentials might explode
                - you can do that by subtracting the minimum
            - the default is `nothing`
                - will use a `eachindex(y)` to generate `x`            
        - `return_history`
            - `Bool`, optional
            - whether to return the complete prediction history or just the final prediction
            - if `false`
                - will return `NTuple{4,Vector{Vector}}`
                    - each entry has `length == 1`
                        - only last element contained
            - if `true`
                - each entry has
                    - `length == n2pred`
            - the default is `false`
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`

    Raises
    ------
        - `AssertionError`
            - if the model has not been fit yet

    Returns
    -------
        - `x_pred_history`
            - `Vector{Vector}`
            - x-values of the extrapolated input
            - returned for `n2pred` iterations if `return_history == true`
            - returned for last iteration if `return_history == false`
        - `y_pred_history`
            - `Vector{Vector}`
            - y-values of the extrapolated input
            - returned for `n2pred` iterations if `return_history == true`
            - returned for last iteration if `return_history == false`
        - `x_history`
            - `Vector{Vector}`
            - x-values of the input
            - returned for `n2pred` iterations if `return_history == true`
            - returned for last iteration if `return_history == false`
        - `y_history`
            - `Vector{Vector}`
            - y-values of the input
            - returned for `n2pred` iterations if `return_history == true`
            - returned for last iteration if `return_history == false`

    Comments
    --------
"""
function predict(
    rpr::RecursivePolynomialRegressor,
    y::Vector, n2pred::Int;
    x::Union{Vector,Nothing}=nothing,
    return_history::Bool=false,
    verbose::Int=0,
    )::NTuple{4, Vector{Vector}}

    @assert rpr.state == :fitted "`rpr` has not been fitted yet. make sure to call `fit(rpr,...)` before predicting!"

    #default parameters
    x       = isnothing(x) ? eachindex(y) : x
    dx      = isnothing(rpr.dx) ? nanmedian(diff(x)) : rpr.dx    #determine stepsize
    nhist   = return_history ? n2pred : 1

    #remove `NaN`
    nanmask = isfinite.(x) .& isfinite.(y)
    x = x[nanmask]
    y = y[nanmask]

    #get exponents (for design matrix)
    exponents = reshape((0:rpr.poly_degree), 1, :)

    #for verbosity (plotting)
    x_history = Vector{Vector}(undef, nhist)        #vector of x-value inputs
    y_history = Vector{Vector}(undef, nhist)        #vector of y-value inputs

    #init in and output
    x_in = copy(x)
    y_in = copy(y)
    x_pred_history = Vector{Vector}(undef, nhist)   #vector of x-value predictions
    y_pred_history = Vector{Vector}(undef, nhist)   #vector of y-value predictions
    for i in 1:n2pred
        
        FormattingUtils.printlog(
            "Iteration $i\n";
            context="predict(RecursivePolynomialRegressor)",
            type=:INFO,
            verbose=verbose,
        )
                
        #get index to save data to (only save history if necessary, otherwise just keep a single entry)
        saveidx = return_history ? i : 1

        #store inputs
        x_history[saveidx] = x_in
        y_history[saveidx] = y_in
        # println(round.(x_history[saveidx]; digits=2))

        #polynomial fit for current subsequence
        x_mat = repeat(x_in, 1, rpr.poly_degree+1) .^ exponents  #design matrix for polynomial regression
        theta_i = x_mat \ y_in

        #update x for next iteration and make prediction
        x_next = vcat(x_in, x_in[end] + dx)  #append next step
        x_mat_next = repeat(x_next, 1, rpr.poly_degree+1) .^ exponents   #design matrix for next iteration
        y_next = x_mat_next * theta_i
        # println(round.(x_next; digits=2))
        
        #store results
        ##NOTE: select correct starting point
        #   - whole series if shorter than `n2pred`
        #   - only last `n2pred` entries if longer or equal length
        startidx = length(x_next)-1 < n2pred ? 1 : (length(x_next)+1) - n2pred  
        x_pred_history[saveidx] = x_next[startidx:end]
        y_pred_history[saveidx] = y_next[startidx:end]

        #update inputs
        startidx = length(x_in) < n2pred ? max(1, (length(x_next)-n2pred)) : 2
        x_in = vcat(x_in[startidx:end], x_next[end])
        y_in = vcat(y_in[startidx:end], y_next[end])

    end

    return x_pred_history, y_pred_history, x_history, y_history
end

function predict(    #signature variation
    rpr::RecursivePolynomialRegressor,
    x::Vector, y::Vector,
    n2pred::Int;
    return_history::Bool=false,
    verbose::Int=0,
    )::NTuple{4, Vector{Vector}}
        
    return predict(rpr, y, n2pred; x=x, return_history=return_history, verbose=verbose)
end

"""
    - extensions to `Plots.plot!()` and `Plots.plot()`
    - plots 2d projection of the result obtained using `Bz3Interp`

    Parameters
    ----------
        - `rpr`
            - `RecursivePolynomialRegressor`
            - fitted instance of `RecursivePolynomialRegressor`
        - `x_pred_history`
            - `Vector{Vector}`
            - prediction of `rpr`
            - x-values of the extrapolated input
        - `y_pred_history`
            - `Vector{Vector}`
            - prediction of `rpr`
            - y-values of the extrapolated input
        - `x_history`
            - `Vector{Vector}`
            - input history of `rpr`
            - x-values of the input
        - `y_history`
            - `Vector{Vector}`
            - input history of `rpr`
            - y-values of the input
        - `dynamic_limits`
            - `Bool`, optional
            - whether to dynamically adjust axis limits as predictions proceed
            - the default is `true`

    Raises
    ------
        - `AssertionError`
            - if the model has not been fit yet

    Returns
    -------
    - `anim`
        - `Plots.Animation`
        - created animation

    Comments
    --------
"""
function Plots.plot(
    rpr::RecursivePolynomialRegressor,
    x_pred_history::Vector{Vector}, y_pred_history::Vector{Vector},
    x_history::Vector{Vector}, y_history::Vector{Vector};
    dynamic_limits::Bool=true,
    )::Plots.Animation

    @assert rpr.state == :fitted "`rpr` has not been fitted yet. make sure to call `fit(rpr,...)` before predicting"

    anim = @animate for i in eachindex(x_pred_history)

        tit = length(y_pred_history) > 1 ? "Iteration $i" : "Final Iteration"

        inlab   = "Current Input"
        polylab = "Polynomial Fit"
        predlab = "Current Prediction"
        plt = scatter(x_history[1], y_history[1];
            label="",
            mc=1, ma=0.3,
            xlabel="x", ylabel="y",
            title=tit,
        )
        if ~dynamic_limits            
            xlims!(
                minimum([minimum.(x_pred_history)..., minimum.(x_history)...]),
                maximum([maximum.(x_pred_history)..., maximum.(x_history)...]),
            )
            ylims!(
                minimum([minimum.(y_pred_history)..., minimum.(y_history)...]),
                maximum([maximum.(y_pred_history)..., maximum.(y_history)...]),
            )
        end
        scatter!(plt, [NaN], [NaN]; label="Past Inputs", mc=1, ma=0.3)
        scatter!(plt, x_history[i],             y_history[i];            mc=2, label=inlab)
        plot!(   plt, x_pred_history[i,:],      y_pred_history[i,:];     lc=3, label=polylab,)
        scatter!(plt, [x_pred_history[i][end]], [y_pred_history[i][end]];mc=3, label=predlab)

    end

    return anim
end

end #module

