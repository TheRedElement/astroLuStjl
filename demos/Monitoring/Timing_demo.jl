
#%%imports
using DataFrames
using Revise

using astroLuStjl.Monitoring.Timing
using astroLuStjl.Styles.FormattingUtils

#%%definitions
function testfunc(x)
    sleep(.1)
    return x^2
end

#%%demos
begin #examples

    x = 1:10
    r1, stats1 = Timing.@exectimer testfunc.(x) "Exec$(1) - Start"
    r2, stats2 = Timing.@exectimer testfunc.(x.+1) "" "Exec2 - End"
    println(r1)

    df = DataFrame([stats1,stats2])
    println(df)
end
