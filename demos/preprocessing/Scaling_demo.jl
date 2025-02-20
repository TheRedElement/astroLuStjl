
#%%imports
using Plots
using Revise

include(joinpath(@__DIR__, "../../src/preprocessing/Scaling.jl"))
include(joinpath(@__DIR__, "../../src/styles/PlotStyleLuSt.jl"))
using .Scaling
using .PlotStyleLuSt

theme(:tre_dark)

#%%definitions

#%%demos
begin #generate some data
    x = sort(rand(-5:5, 100))
    # ]
    y = x .^2 .+ .+ 3randn(size(x))
end
begin #MinMaxScaler
    mms = Scaling.MinMaxScaler(0, 1)
    mms = Scaling.fit(mms, y)
    y_scaled = Scaling.transform(mms, y)

    Scaling.plot(mms, y)

end