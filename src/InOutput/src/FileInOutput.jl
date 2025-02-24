
#TODO: implement examples

"""
    - module implementing methods to load different file structures

    Structs
    -------

    Functions
    ---------
        - `read_psv()`

    Extended Functions
    ------------------

    Dependencies
    ------------
        - `CSV`
        - `DataFrames`

    Comments
    --------

    Examples
    --------
        - see [FileInOutput_demo.jl](../../demos/InOutput/FileInOutput_demo.jl)

"""

module FileInOutput

#%%imports
using CSV
using DataFrames

#import for extending

#intradependencies

#%%exports
export read_psv

#%%definitions
"""
    - reader for "Pattern Separated Value"-files (PSV)
    - wrapper function around `CSV.File()`
    - enables the user to load file with arbitrary separation pattern (defined by `seq`)

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

end #module


