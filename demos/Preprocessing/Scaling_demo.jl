
#%%imports
using Plots
using Revise

using astroLuStjl.Preprocessing.Scaling
using astroLuStjl.Styles.PlotStyleLuSt
using astroLuStjl.Styles.FormattingUtils

theme(:tre_dark)

#%%definitions

#%%demos
begin #generate some data
    x  = sort(rand(-5:5, 100))
    x2 = ones(100)              #only one unique value (edge case)

    y = x .^2 .+ .+ 3randn(size(x))
    y2 = x2
end
begin #MinMaxScaler
    #standard application
    mms = Scaling.MinMaxScaler(0, 1)
    mms = Scaling.fit(mms, y)
    y_scaled = Scaling.transform(mms, y)

    Scaling.plot(mms, y)
    
    #one unique value (edge case)
    mms = Scaling.MinMaxScaler(0, 1)
    mms = Scaling.fit(mms, y2)
    y_scaled = Scaling.transform(mms, y2)

    Scaling.plot(mms, y2)

end
