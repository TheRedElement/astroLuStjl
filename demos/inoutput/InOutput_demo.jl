
#%%imports
using Revise

include(joinpath(@__DIR__, "../../src/inoutput/InOutput.jl"))
using .InOutput



begin#examples
    fname = joinpath(@__DIR__, "../_data/psv.csv")
    df = InOutput.read_psv(fname; sep=r"\ +", csv_file_kwargs=Dict(:comment=>"#", :header=>true, :normalizenames=>true))
    display(df)
end