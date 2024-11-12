
#%%imports
using Revise

include(joinpath(@__DIR__, "../src/SigmaClipping.jl"))
include(joinpath(@__DIR__, "../src/PlotStyleLuSt.jl"))

using .PlotStyleLuSt
# theme(:tre_dark)
theme(:tre_light)

#%%generate data
x = [
    sort(rand(-10:0, 100)),
    collect(range(0,10, 500)),
]
y = map(xk -> xk .^2 .+ 3randn(size(xk)), x)
y[2] .+= rand([1,0,0,0,0,0], size(y[2])) .* rand(-50:50, size(y[2]))

#%%init transformer
sc_bng = [
    SigmaClipping(b, b;
        max_iter=3,
        sigma_l_decay=.8, sigma_u_decay=.9,
        method=:binning,
    ) for b in [1.5,2]
]
sc_poly = [
    SigmaClipping(b, b;
        max_iter=5,
        sigma_l_decay=.8, sigma_u_decay=.9,
        method=:poly,
    ) for b in [0.5,1.0]
]

#%%`SigmaClipping` using `Binning`
scr = fit.(sc_bng, x, y;
    generate_bins=generate_bins_pts,
    y_start=nothing, y_end=nothing,
    verbose=2,
    generate_bins_args=(0.1,),
    generate_bins_kwargs=(eps=1e-3, verbose=0),
)
scr = fit.(sc_bng, x, y;
    generate_bins=generate_bins_int,
    y_start=nothing, y_end=nothing,
    verbose=2,
    generate_bins_args=(20,),
    generate_bins_kwargs=(eps=1e-3, verbose=0),
)

x_clipped, y_clipped = transform.(scr, x, y; verbose=0)

p = plot(
    plot(scr[1]; x=x[1], y=y[1], sc=sc_bng[1], show_clipped=true),
    plot(scr[2]; x=x[2], y=y[2], sc=sc_bng[2], show_clipped=true),
    layout=(1,2),
)
display(p)


#%%`SigmaClipping` using polynomials
scr = fit.(sc_poly, x, y;
    p_deg=2,
    verbose=2,
)
x_clipped, y_clipped = transform.(scr, x, y; verbose=0)

p = plot(
    plot(scr[1]; x=x[1], y=y[1], sc=sc_poly[1], show_clipped=true),
    plot(scr[2]; x=x[2], y=y[2], sc=sc_poly[2], show_clipped=true),
    layout=(1,2),
)
display(p)
