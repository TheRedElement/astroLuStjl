
#%%imports
using LaTeXStrings
using Plots
using Random
using Revise
using StatsBase
# Random.seed!(1)

using astroLuStjl.Datascience.TimeseriesForecasting
using astroLuStjl.Preprocessing.Preprocessing
using astroLuStjl.Styles.FormattingUtils
using astroLuStjl.Styles.PlotStyleLuSt

theme(:tre_dark)

#%%definitions

#%%demos
begin #RecursivePolynomialRegressor
    #%%generate some data
    deg = 8
    x1 = collect(range(0, 1, 101))          #trivial example
    x2 = collect(range(1000, 1500, 101))    #much higher x-values => exploding exponents

    #y is the same for both `x`
    y = mapslices(xi -> xi.^(0:deg), reshape(x1, :, 1); dims=2)
    y = y * rand(-4:.1:4, deg+1) .+ 0.2*randn(size(y,1))

    #split in train and test set
    ntrain = 50
    x_tr1   = x1[1:ntrain]
    x_tr2   = x2[1:ntrain]
    x_te1   = x1[(ntrain+1):end]
    x_te2   = x2[(ntrain+1):end]
    y_tr    = y[1:ntrain]
    y_te    = y[(ntrain+1):end]

    # #add `NaN`
    # y_tr[3] = NaN
    # x_tr1[10:20] .= NaN

    begin #case 1: trivial
        n2pred = 40

        #fit the predictor
        rpr1 = TimeseriesForecasting.RecursivePolynomialRegressor(3; dx=0.02) #instantiate regressor
        println(rpr1)

        rpr1 = TimeseriesForecasting.fit(rpr1, x_tr1, y_tr; verbose=2)  #fit
        x_pred_hist, y_pred_hist, x_hist, y_hist = TimeseriesForecasting.predict(rpr1, x_tr1, y_tr, n2pred; return_history=true, verbose=2) #predict

        #plot result
        anim = plot(rpr1,
            x_pred_hist, y_pred_hist,
            x_hist, y_hist;
            dynamic_limits=true
        )
        display(gif(anim, fps=10))

    end

    begin #case 2: preprocessing required (exploding exponentials)
        deg = 5
        n2pred = 20
        #without processing
        rpr2_1 = TimeseriesForecasting.RecursivePolynomialRegressor(deg)
        x_pred_hist1, y_pred_hist1, x_hist1, y_hist1 = TimeseriesForecasting.predict(rpr2_1, x_tr2, y_tr, n2pred) #predict

        #with processing
        rpr2_2 = TimeseriesForecasting.RecursivePolynomialRegressor(deg)
        x_tr2_norm = x_tr2 .- minimum(x_tr2)
        x_pred_hist2, y_pred_hist2, x_hist2, y_hist2 = TimeseriesForecasting.predict(rpr2_2, x_tr2_norm, y_tr, n2pred) #predict

        #since `save_history == false`
        x_pred1 = x_pred_hist1[1]
        y_pred1 = y_pred_hist1[1]
        x_pred2 = x_pred_hist2[1]
        y_pred2 = y_pred_hist2[1]

        #plot result
        p1 = scatter(x_tr2, y_tr; label="Train Set")
        scatter!(p1, x_te2, y_te; label="Test Set")
        plot!(p1, x_pred1, y_pred1; ls=:solid, label="With Processing")
        plot!(p1, x_pred1, y_pred2; ls=:dash,  label="Without Processing")  #using same x-values to show in same panel

        p = plot(p1; xlabel="x", ylabel="y")
        display(p)
    end

    begin #case 3: too few datapoints (use first few predictions to extrapolate further)
        n2pred = 50
        ntrain = 20   #`nsamples < n2pred`
        x_tr3 = x1[1:ntrain]
        y_tr3 = y[1:ntrain]
        x_va3 = x1[(ntrain+1):end]
        y_va3 = y[(ntrain+1):end]

        rpr3 = TimeseriesForecasting.RecursivePolynomialRegressor(2;) #instantiate regressor
        
        x_pred, y_pred, x_hist, y_hist = TimeseriesForecasting.predict(rpr3, x_tr3, y_tr3, n2pred; return_history=true) #predict

        #plot result
        anim = plot(rpr3, x_pred, y_pred, x_hist, y_hist; dynamic_limits=false)
        display(gif(anim, fps=10))

    end
end

begin #ARIMA
    begin #generating data
        ntrain = 14
        ntest = 10
        ntimes = ntrain+ntest
        
        #filter
        h1 = [0.5]          #AR(1) process (Box1995 p. 151)
        h2 = [0.75;-0.5]    #AR(2) process (Box1995 p. 151)
        
        #autoregressively generate
        # u = ar_process(ntimes, h1; offset=1.0, var=0.01)    #AR(1)
        u = TimeseriesForecasting.ar_process(ntimes, h2; offset=1.0, var=100)    #AR(2)
        
        #add trends
        t = axes(u,1).-1
        # u .+= (t ./ 10) .^ 1                #linear
        # u .+= (t ./ 1) .^ 2                 #quadratic
        u_trend = u .+ -(t ./ 1) .^ 2 + 0.1*t .^ 3    #cubic
        
        #split into train and test set
        u_tr = u[1:ntrain]
        u_te = u[ntrain+1:end]
        u_trend_tr = u_trend[1:ntrain]
        u_trend_te = u_trend[ntrain+1:end]
        

    end

    begin #AutoRegressor
        ar = TimeseriesForecasting.AutoRegressor(2)
        ar = TimeseriesForecasting.fit(ar, u_tr)
        x_pred_ar = TimeseriesForecasting.predict(ar, u_tr, ntest; verbose=0)
        p = plot(ar, u_tr)
        plot!(p, ntrain .+ axes(u_te,1), x_pred_ar; lc=1, ls=:dash, label="AR Forecast")
        plot!(p, ntrain .+ axes(u_te,1), u_te; label="Ground Truth")
        display(p)
    end

    begin #MovingAverage
        
        ma = TimeseriesForecasting.MovingAverage(5)
        ma = TimeseriesForecasting.fit(ma, ar.residuals)
        x_pred_ma = TimeseriesForecasting.predict(ma, ar.residuals, ntest; verbose=0)
        p = plot(ma, ar.residuals, u_tr)
        plot!(p, ntrain .+ axes(u_te,1), x_pred_ma; lc=1, ls=:dash, label="MA Forecast")
        plot!(p, ntrain .+ axes(u_te,1), u_te; lc=4, label="Ground Truth")
        # #adding AR fit (residuals based on AR)
        # ar_fit = Preprocessing.get_subsequences(u_tr, ar.p.+1; dim=1)' * ar.phi
        # plot!(p, axes(u_tr,1)[ar.p+1:end], ar_fit, ls=:dashdot, label="AR Prediction")
        display(p)
    end

    begin #ARIMA
        arima = TimeseriesForecasting.ARIMA(3,2,5)
        println(arima)
        arima = TimeseriesForecasting.fit(arima, u_trend_tr; verbose=0)
        # println(arima)

        x_pred_arima = TimeseriesForecasting.predict(arima, u_trend_tr, ntest; verbose=0)
        
        p = plot(arima, u_trend_tr; n2pred=ntest)
        plot!(p, ntrain .+ axes(u_trend_te,1), u_trend_te; label="Ground Truth")
        display(p)
    end
    
end
