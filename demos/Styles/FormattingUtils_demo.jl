
#%%imports
using Plots
using Revise

using astroLuStjl.Styles.FormattingUtils
using astroLuStjl.Styles.PlotStyleLuSt

theme(:tre_dark)

#%%definitions

#%%demos
begin #examples

    #passband color sequence
    x = range(1,size(FormattingUtils.pb_colors, 1))'
    display(heatmap(x;
        size=(900,200),
        color=cgrad(FormattingUtils.pb_colors; categorical=true),
        xticks=(x[1,:],["fuv", "nuv", "b", "g", "y", "r", "nir", "fir"]), yticks=false,
        xlabel="Passband",
        bottom_margin=10Plots.mm,
        cbar=false,
        
    ))

    #formatted printing
    FormattingUtils.printlog(
        "This is a test",
        context=nothing,
        type=:INFO,
        level=2,
        start=">"^4,
        verbose=2,
        printstyled_kwargs=Dict(:color=>:blue),
    )
end