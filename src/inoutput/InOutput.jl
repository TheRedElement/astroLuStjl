
"""
    - script containing functions to load different file structures

    Dependencies
    ------------
        - `CSV`
        - `DataFrames`

    Comments
    --------
"""


#%%imports
using CSV
using DataFrames

#%%definitions
"""
    - reader for "Pattern Separated Value"-files (PSV)
    - wrapper function around `CSV.File()`
    - enables the user to load file with arbitrary separation pattern (defined by `Regex`)

    Parameters
    ----------
        - `filename`
            - `String`
            - path to the file to extract
        - `sep`
            - `Regex`, optional
            - regular expression defining the separator pattern
            - everything matching `sep` will be considered a separator
            - the default is `r","`
                - equivalent to a csv-file
        - `csv_file_kwargs`
            - `Dict`, optional
            - kwargs to pass to `CSV.File()`

    Raises
    ------

    Returns
    -------
        - `df`
            - `DataFrame`
            - the input file read into a `DataFrame`

    Dependencies
    ------------
        - `CSV`
        - `DataFrames`

    Comments
    --------
"""
function read_psv(
    filename::String;
    sep::Regex=r",",
    csv_file_kwargs::Dict{Symbol,Any}=Dict()
    )::DataFrame

    #read file
    f = read(filename, String)

    #remove leading and trailing sep-characters
    f = replace(f, Regex("^"*sep.pattern, "m")=>"")
    f = replace(f, Regex(sep.pattern*"\$", "m")=>"")
    
    #replace `sep` with comma (to make compatible with CSV)
    f = replace(f, sep=>",")
    
    df = CSV.File(IOBuffer(f); csv_file_kwargs...) |> DataFrame
    return df
end

# #examples
# using Glob

# ####################################
# #SDSS
# fnames = Glob.glob("../data/sdss_sn_lcs/*.sum", @__DIR__)
# # println(fnames)
# println(fnames[1])

# df = read_psv(fnames[1]; sep=r"\ +", csv_file_kwargs=Dict(:comment=>"#", :header=>false, :normalizenames=>true))
# #get header (last comment `#``)
# header = readlines(fnames[1])
# header = lowercase.(split(filter(contains(r"^#"), header)[end], r"\s+"))
# header[1] = header[1][2:end]
# rename!(df, header)
# display(df)


# #####################################
# #TESS
# fnames = Glob.glob("../data/tess_sn_lcs/*.txt", @__DIR__)
# # println(fnames)
# println(fnames[1])

# df = read_psv(fnames[1]; sep=r"\ +", csv_file_kwargs=Dict(:comment=>"#", :header=>false, :normalizenames=>true))
# #get header (last comment `#``)
# header = readlines(fnames[1])
# header = lowercase.(split(filter(contains(r"^#"), header)[end], r"\s+"))
# println(header)
# rename!(df, header[2:end])
# display(df)
