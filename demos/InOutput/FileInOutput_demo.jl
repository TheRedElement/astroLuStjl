
#%%imports
using Revise

using astroLuStjl.InOutput.FileInOutput
using astroLuStjl.Styles.FormattingUtils

#%%definitions


#%%demos
begin #examples
    fname = joinpath(@__DIR__, "../_data/psv.csv")
    df = FileInOutput.read_psv(fname; sep=r"\ +", csv_file_kwargs=Dict(:comment=>"#", :header=>true, :normalizenames=>true))
    display(df)
end
