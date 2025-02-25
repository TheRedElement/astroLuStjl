
#%%imports
using Revise

using astroLuStjl.Styles.FormattingUtils

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