
#%%imports
using Plots
using Revise

include(joinpath(@__DIR__, "../../src/preprocessing/DataBinning.jl"))
include(joinpath(@__DIR__, "../../src/preprocessing/OutlierRemoval.jl"))
include(joinpath(@__DIR__, "../../src/styles/PlotStyleLuSt.jl"))
using .DataBinning
using .OutlierRemoval
using .PlotStyleLuSt
theme(:tre_dark)

#%%definitions

#%%demos
begin #SigmaClipping
    #generate data
    x = [
        sort(rand(-10:0, 100)),
        collect(range(0,10, 500)),
    ]
    y = map(xk -> xk .^2 .+ 3randn(size(xk)), x)
    y[2] .+= rand([1,0,0,0,0,0], size(y[2])) .* rand(-50:50, size(y[2]))

    #init transformer
    sc_bng = [
        OutlierRemoval.SigmaClipping(b, b;
            max_iter=3,
            sigma_l_decay=.8, sigma_u_decay=.9,
            method=:binning,
        ) for b in [1.5,2]
    ]
    sc_poly = [
        OutlierRemoval.SigmaClipping(b, b;
            max_iter=5,
            sigma_l_decay=.8, sigma_u_decay=.9,
            method=:poly,
        ) for b in [0.5,1.0]
    ]

    #`SigmaClipping` using `DataBinning`
    scr = OutlierRemoval.fit.(sc_bng, x, y;
        generate_bins=DataBinning.generate_bins_pts,
        y_start=nothing, y_end=nothing,
        verbose=2,
        generate_bins_args=(0.1,),
        generate_bins_kwargs=(eps=1e-3, verbose=0),
    )
    scr = OutlierRemoval.fit.(sc_bng, x, y;
        generate_bins=DataBinning.generate_bins_int,
        y_start=nothing, y_end=nothing,
        verbose=2,
        generate_bins_args=(20,),
        generate_bins_kwargs=(eps=1e-3, verbose=0),
    )

    x_clipped, y_clipped = OutlierRemoval.transform.(scr, x, y; verbose=0)

    p = plot(
        OutlierRemoval.plot(scr[1]; x=x[1], y=y[1], sc=sc_bng[1], show_clipped=true),
        OutlierRemoval.plot(scr[2]; x=x[2], y=y[2], sc=sc_bng[2], show_clipped=true),
        layout=(1,2),
    )
    display(p)


    #`SigmaClipping` using polynomials
    scr = OutlierRemoval.fit.(sc_poly, x, y;
        p_deg=2,
        verbose=2,
    )
    x_clipped, y_clipped = OutlierRemoval.transform.(scr, x, y; verbose=0)

    p = plot(
        OutlierRemoval.plot(scr[1]; x=x[1], y=y[1], sc=sc_poly[1], show_clipped=true),
        OutlierRemoval.plot(scr[2]; x=x[2], y=y[2], sc=sc_poly[2], show_clipped=true),
        layout=(1,2),
    )
    display(p)
end