#TODO: not working for some reason!

"""
    - module implementing functionalities for monitoring executions of scripts, functions etc.

    Structs
    -------
    
    Functions
    ---------
        - `@exectimer()`

    Dependencies
    ------------
        - `Dates`
    
    Comments
    --------

    Examples
    --------
        - see [Monitoring_demo.jl](../../demos/monitoring/Monitoring_demo.jl)    
"""

module Monitoring

#%%imports
using Dates

#import for extending

#intradependencies
include(joinpath(@__DIR__,"../monitoring/FormattingUtils.jl"))
using .FormattingUtils

#%%exports
export exectimer

FormattingUtils.printlog(
    "This module is currently not working! Do not use!\n",
    context="`Monitoring.jl`",
    type=:WARNING,
    verbose=1,
    printstyled_kwargs=Dict(:color=>:yellow)
)

#%%definitions
"""
    - wrapper around `@timed` to
        - return result after expression evaluation
        - time expression evaluation and return statistics as `NamedTuple`
            - can be easily converted to `DataFrames.DataFrame`
    
    Parameters
    ----------
        - `expr`
            - `Expr`
            - expression to be timed and executed
        - `comment_start`
            - `String`, optional
            - a comment to add at the verbosity at the start of the expression evaluation
            - useful for i.e. tagging in the dataframe
            - the default is `""`
        - `comment_end`
            - `String`, optional
            - a comment to add at the verbosity at the end of the expression evaluation
            - useful for i.e. tagging in the dataframe
            - the default is `""`
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`

    Raises
    ------

    Returns
    -------
        - `expr_res`
            - `Any`
            - result of the evaluation of `expr`
        - `exec_stats`
            - `NamedTuple`
            - contains evaluation statistics of the expression

    Dependencies
    ------------
        - `Dates`

    Comments
    --------
"""
macro exectimer(
    expr::Expr,
    comment_start::Union{String,Expr}="", comment_end::Union{String,Expr}="",
    verbose=0
    )
    #expression charateristics
    name = expr.args[1]
    return quote
        #make local variables accessible to quote
        name            = $name
        comment_start   = $comment_start
        comment_end     = $comment_end

        t_start = now()         #get starting time of expression evaluation
        println("Started  $name() at $t_start. $(eval(comment_start))")
        
        #evaluate expression `expr`
        res = @timed $expr
        
        t_end = now()           #get end time of expression evaluation
        
        println("Finished $name() at $t_end. Elapsed time: $(res.time) s. $(eval(comment_end))")

        expr_res    = res.value     #expression output
        exec_stats  = (             #timing statistics
            name=$name,
            t_start=t_start,
            t_end=t_end,
            duration_seconds=res.time,
            comment_start=$comment_start,
            comment_end=$comment_end,
        )

        #`quote output` = `macro output`
        (expr_res, exec_stats)
    end
end

end #module

