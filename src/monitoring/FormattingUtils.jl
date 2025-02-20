
#TODO: docstrings + formatting of code

"""
    - module implementing methods for custom formatting

    Structs
    -------

    Functions
    ---------
        - `printlog()`

    Extended Functions
    ------------------

    Dependencies
    ------------
    
    Comments
    --------

    Examples
    --------
        - see [FormattingUtils_demo.jl](../../demos/monitoring/FormattingUtils_demo.jl)
"""

module FormattingUtils

#%%imports

#import for extending

#intradependencies

#%%exports
export printlog

"""
    - function to print a formatted logging message

    Parameters
    ----------
        - `msg`
            - `String`
            - message to be printed
        - `context`
            - `String`, `Nothing`, optional
            - context to the printed message
            - the default is `nothing`
                - will print `''`
        - `type`
            - `Symbol`, optional
            - type of the message
            - allowed `Symbol`s are
                - `:INFO`
                - `:WARNING`
            - the default is `:INFO`
        - `level`
            - `Int`, optional
            - level of the message
            - will append `repeat(start,level)` at the start of the message
                - i.e., indent the message
            - the default is 0
                - no indentation
        - `start`
            - `String`, `Nothing`, optional
            - string used to mark levels
            - will print `repeat(start,level)` before `msg`
            - the default is `nothing`
                - will be set to `"    "`
                - i.e., 4 spaces
                - do not  use something like `" "^4`, since that can sometimes be interpreted as an integer!
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`
        - `printstyled_kwargs`
            - `Dict`, optional
            - kwargs to pass to `printstyled()`
            - the default is `nothing`
                - will be set to `Dict()`

    Raises
    ------
        - `MethodError: no method matching repeat(::UInt8, ::Int64)`
            - sometimes raised when passing something like `start=" "^4`
            - not known why this is the case

    Returns
    -------

    Dependencies
    ------------

    Comments
    --------
        - passing `start` for example in the form of `start=" "^4` can sometimes lead to an error for some reason
"""
function printlog(
    msg::String;
    context::Union{String,Nothing}=nothing,
    type::Symbol=:INFO,
    level::Int=0,
    start::Union{String,Nothing}=nothing,
    verbose::Int=0,
    printstyled_kwargs::Union{Dict,Nothing}=nothing,
    )::Nothing

    if !(type in [:INFO :WARNING])
        throw(ArgumentError("`type` has to be in $([:INFO :WARNING])!"))
    end

    #default parameters
    context             = isnothing(context) ? "" : context
    start               = isnothing(start) ? "    " : start
    printstyled_kwargs  = isnothing(printstyled_kwargs) ? Dict() : printstyled_kwargs

    vbs_th = (type == :INFO) ? 2 : (type == :WARNING ? 1 : 0)

    to_print = "$(repeat(start,level))$type($context): $msg"
    
    if verbose >= vbs_th
        printstyled(to_print; printstyled_kwargs...)
    end
    
end


end #module
