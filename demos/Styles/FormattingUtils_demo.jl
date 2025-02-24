
#%%imports
using Revise

include(joinpath(@__DIR__, "../../src/monitoring/FormattingUtils.jl"))
using .FormattingUtils

#%%definitions

#%%demos
begin #examples

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