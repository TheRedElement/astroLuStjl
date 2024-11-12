
#%%imports
using Dates
using Plots
using Revise


#custom modules
include(joinpath(@__DIR__, "../src/StringParsing.jl"))
using .StringParsing

#%%demos
begin ##parsing `CompoundPeriod`
    cp = Dates.CompoundPeriod(Hour(1), Minute(42), Second(24), Millisecond(64))
    cp_s = "$cp"
    println(cp_s)
    # s = "1 hour, 42 minutes, 24 seconds, 64 milliseconds"   #test string

    cp = StringParsing.parse2compoundperiod(cp_s)
    println(cp)

    #convert to hours
    hours = Dates.value(sum(Millisecond, cp.periods)) / 1000 / 60 / 60
    println("$hours hours")
end
