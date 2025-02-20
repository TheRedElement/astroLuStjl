#%%imports
using CategoricalArrays
using DataFrames
using Distributions
using NaNStatistics
using Plots
using PGFPlotsX
using Random
using Revise

#custom modules
include(joinpath(@__DIR__, "../../src/preprocessing/Bezier3Interp.jl"))
include(joinpath(@__DIR__, "../../src/styles/PlotStyleLuSt.jl"))
include(joinpath(@__DIR__, "../../src/visualization/PlotTypes.jl"))
include(joinpath(@__DIR__, "../../src/preprocessing/Scaling.jl"))
using .Bezier3Interp
using .PlotStyleLuSt
using .PlotTypes
using .Scaling

theme(:tre_dark)

gr()
# plotlyjs()
# pgfplotsx()   #supports latex output

#%%definitions

#%%demos
begin #hatched histograms
    hg1 = PlotTypes.hatched_histogram(
        randn(300);
        k=-5, offset=2, npoints=50, ls=:dash,
        normalize=:false, fillalpha=.8, fillcolor=1
    )

    x = randn(300)
    p2 = plot(-3:.1:3, pdf(Normal(0,1), -3:.1:3); label="")
    # p3 = plot()
    PlotTypes.hatched_histogram!(
        p2, x;
        k=-0.1, offset=0.02, ls=:dashdot,
        normalize=:true, fillalpha=0.0
    )
    PlotTypes.hatched_histogram!(
        p2, x;
        k=0.1, offset=0.02, ls=:dashdot,
        normalize=:true, fillalpha=0.0,
    )

    display(plot(hg1, p2; layout=(1,2), xlabel="x", ylabel="y", size=(1200,400)))
end

begin #parallel coordinates
    #defining custom preamble to ensure working with `PGFPlotsX`
    PGFPlotsX.CUSTOM_PREAMBLE = ["\\usepackage{pmboxdraw}", "\\usepackage{graphicx}", "\\newcommand{\\blockfull}{\\scalebox{.4}[1]{\\textblock}}"]

    # Random.seed!(1)

    nsamples = 100
    df = DataFrames.DataFrame(
        :x1     => rand([1,50,100], nsamples),
        :x2     => rand(["x2_1", "x2_2"], nsamples),
        :x3     => randn(nsamples),
        :x4     => repeat([-10], nsamples),
        :score  => sort(rand(0:.01:10, nsamples)),
    )
    p = PlotTypes.parallel_coords(
        df;
        res=30,
        slopes_min=.5, slopes_max=2,
        cmap=:grays,
        #yticks options
        n_yticks=3,
        ytickfontsize=nothing, yrotation=-10.,
        ytickcolor=:cyan, yticklabelcolor=RGB(1,0,0), bg=RGBA(0,0,0,1),
        ytickalpha=1., 
        yticksize=1.,
        #additional kwargs
        bottom_margin=12Plots.mm,
        left_margin=10Plots.mm, right_margin=10Plots.mm,
        linealpha=0.8,
        xrotation=50,
        size=(900,400),
        xlabel="Coordinates", ylabel="Values",
    )
    #adding the best model on top
    df_num = select(df, :, names(df, AbstractString) .=> x -> levelcode.(categorical(x)), renamecols=false)
    df_scaled = mapcols(Scaling.minmaxscaler, df_num)
    # xx_best, yy_best = PlotTypes.pc_bezier3_smoothing(Matrix(df_scaled[end:end,:]), yspread=0.02, dyspread=3, bzfreedom=0.2)
    xx_best, yy_best = PlotTypes.pc_sigmoid_itp(Matrix(df_scaled[end:end,:]), res=30, slopes_min=1, slopes_max=1)
    
    plot!(p,
        xx_best, yy_best';
        linecolor=Plots.default(:palette)[1],
        label="Best Model", legend=true,
    )
    display(p)
end


