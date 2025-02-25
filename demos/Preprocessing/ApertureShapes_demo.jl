
#%%imports
using Plots
using Revise

#custom modules
using astroLuStjl.Preprocessing.ApertureShapes
using astroLuStjl.Styles.PlotStyleLuSt
using astroLuStjl.Styles.FormattingUtils

theme(:tre_dark)

gr()

#%%definitions

#%%demos

begin #general way of constructing apertures
    #pseudo data
    frame = rand(-0.5:0.5, (20,15))
    xpix, ypix = size(frame)
    y2plot = 1:xpix
    x2plot = 1:ypix

    #construting apertures
    apt = ApertureShapes.ApertureTemplate(xpix, ypix, 2, -3; normalize=:npixels, npixels=1, outside=NaN)  #aperture generator

    ap1_mask = ApertureShapes.gauss_aperture(apt, 2, 2) #actual generation
    ap2_mask = ApertureShapes.lp_aperture(apt, 8, 1; radius_inner=5) #actual generation

    ApertureShapes.print_aperture(ap2_mask)
end

begin #general way of plotting apertures
    p = heatmap(x2plot,y2plot,frame; colorbar=false, aspect_ratio=:equal, clim=(-2,2))
    ApertureShapes.plot_aperture!(p, ap1_mask, y2plot, x2plot; linecolor=:white, linealpha=:ap_mask, lw=5)    #outline
    ApertureShapes.plot_aperture!(p, ap1_mask, y2plot, x2plot; fillcolor=1, fillalpha=:line)
    ApertureShapes.plot_aperture!(p, ap2_mask, y2plot, x2plot; linecolor=:white, linealpha=:ap_mask, fillalpha=0, ls=:dash, lw=5)    #outline
    ApertureShapes.plot_aperture!(p, ap2_mask, y2plot, x2plot; linecolor=:blue, fillalpha=.2, ls=:dash, )
    plot!(p, [NaN]; color=:red, label="Aperture")
    plot!(p, [NaN]; color=:blue, ls=:dash, label="Sky-Ring")
    plot!(p; legend_position=:topleft)
    display(p)

    #just the aperture
    p = ApertureShapes.plot_aperture(ap1_mask; fillalpha=:ap_mask, ls=:solid, aspect_ration=:equal)
    display(p)
end

begin #examples for different settings (standard aperture)
    apt = ApertureShapes.ApertureTemplate(50, 60, 0, 0; npixels=1, outside=0)  #aperture generator

    apertures = [
        ApertureShapes.lp_aperture(apt, 10, 1;  poly_coeffs=nothing),
        ApertureShapes.lp_aperture(apt, 10, 1.5;poly_coeffs=nothing),
        ApertureShapes.lp_aperture(apt, 10, 1.5;poly_coeffs=[-0.015, 0.16, 0, 10]),
        ApertureShapes.lp_aperture(apt, 10, 2;  poly_coeffs=[0,1]),
        ApertureShapes.lp_aperture(apt, 10, 2;  poly_coeffs=[-0.015, 0.16, 0, 10]),
        ApertureShapes.lp_aperture(apt, 10, Inf),
        ApertureShapes.rect_aperture(apt, 10, 20),
        ApertureShapes.gauss_aperture(apt, 5, 0;   covariance=[[10. -13.];[-13. 50.]], lp=false),
        ApertureShapes.gauss_aperture(apt, 5, Inf; covariance=5, lp=false),
        ApertureShapes.gauss_aperture(apt, 5, 1;   covariance=5, lp=true),
        ApertureShapes.lorentz_aperture(apt, 20, 30, Inf,    0;),
        ApertureShapes.lorentz_aperture(apt, 10, 10, 10,     Inf;),
    ]

    plots = []
    for idx in eachindex(apertures)
        a = ApertureShapes.plot_aperture(apertures[idx]; linecolor=1, aspect_ratio=:equal)
        plot!(a, [NaN]; linecolor=1, label="Aperture")
        push!(plots, a)
    end

    p = plot(plots...; size=(1200,1200), xlabel="y-pixel", ylabel="x-pixel")
    display(p)
end


begin #examples for different settings (sky-ring)
    apt = ApertureShapes.ApertureTemplate(50, 60, 0, 0; npixels=1, outside=0)  #aperture generator

    apertures = [
        ApertureShapes.lp_aperture(apt, 10, 1;  poly_coeffs=nothing, radius_inner=4),
        ApertureShapes.lp_aperture(apt, 10, 1.5;poly_coeffs=nothing, radius_inner=4),
        ApertureShapes.lp_aperture(apt, 10, 1.5;poly_coeffs=[-0.015, 0.16, 0, 10], radius_inner=4),
        ApertureShapes.lp_aperture(apt, 10, 2;  poly_coeffs=[0,1], radius_inner=4),
        ApertureShapes.lp_aperture(apt, 10, 2;  poly_coeffs=[-0.015, 0.16, 0, 10], radius_inner=4),
        ApertureShapes.lp_aperture(apt, 10, Inf, radius_inner=4),
        ApertureShapes.rect_aperture(apt, 10, 20; width_inner=4, height_inner=10),
        ApertureShapes.gauss_aperture(apt, 5, 0;   covariance=[[10. -13.];[-13. 50.]], lp=false, radius_inner=4),
        ApertureShapes.gauss_aperture(apt, 5, Inf; covariance=5, lp=false,  radius_inner=4),
        ApertureShapes.gauss_aperture(apt, 5, 1;   covariance=5, lp=true,   radius_inner=4),
        ApertureShapes.lorentz_aperture(apt, 20, 30, Inf,    0;     radius_inner=4),
        ApertureShapes.lorentz_aperture(apt, 10, 10, 10,     Inf;   radius_inner=4),
    ]

    plots = []
    for idx in eachindex(apertures)
        a = ApertureShapes.plot_aperture(apertures[idx]; linecolor=1, aspect_ratio=:equal)
        plot!(a, [NaN]; linecolor=1, label="Aperture")
        push!(plots, a)
    end

    p = plot(plots...; size=(1200,1200), xlabel="y-pixel", ylabel="x-pixel")
    display(p)
end
