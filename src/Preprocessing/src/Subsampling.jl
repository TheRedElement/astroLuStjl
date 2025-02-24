"""
    - module implementing functions for data preprocessing

    
    Structs
    -------
    
    Functions
    ---------
        - `get_subsequences()`
        - `split_along_dim()`

    Extended Functions
    ------------------
    
    Dependencies
    ------------
        - `DataFrames`
        - `Random`

    Comments
    --------

    Examples
    --------
        - see [Subsampling_demo.jl](../../demos/preprocessing/Subsampling_demo.jl)

"""

module Subsampling

#%%imports
using DataFrames
using Random

#import for extending

#intradependencies

#%%exports
export get_subsequences
export split_along_dim


#%%definitions
"""
    - functon to extract subsequences along some dimension `dim` of an `AbstractArray` `x`
    - does so via a sliding window of length `seq_len` with a stepsize of `stride`

    Parameters
    ----------
        - `x`
            - `AbstractArray`
            - input dataset to be split into subsequences of size `seq_len`
        - `seq_len`
            - `Int`
            - length each of the extracted subsequences shall have
        - `dim`
            - `Int`, optional
            - dimension to apply the subsequence extraction to
        - `stride`
            - `Int`, optional
            - number of steps the start of two consecutuve subsequences is offset to  each other
            - equivalent to a stesize in a sliding window
            - if you just want to split a series into chunks of length `seq_len`, set `stride` to the same value as `seq_len`
            - the default is `1`
        - `verbose`
            - `Int`, optional
            - verbosity level

    Raises
    ------
        - `AssertionError`
            - if `stride > 1`

    Returns
    -------
        - `out`
            - `AbstractArray`
            - `x` but split into its subsequences
                - the subsequences are stored along a new dimension (last dimension in `out`)
            - has `ndim(x)+1` dimensions

    Dependencies
    ------------

    Comments
    --------

    Examples
    --------
```julia
    #1d series
    x = 0:99
    out = get_subsequences(x, 10; dim=1, stride=5)
    p = plot(x, zeros(size(x)); label="Input")
    plot!(p, [NaN]; color=2, label="Subsequences")
    for i in axes(out,2)
        plot!(p, out[:,i], ones(size(out,1)).+i, color=2, label="")
    end
    display(p)

    #nd series
    x = (1:10:30) * (1:100)'
    display(x)
    out = get_subsequences(x, 30; dim=2, stride=1)
    p = plot(
        plot(x[1,:], ones(size(x,2)), label="Input"),
        plot(x[2,:], ones(size(x,2)), label=""),
        plot(x[3,:], ones(size(x,2)), label="");
        layout=@layout[[a b]; [_ c{0.5w} _]]
    )
    plot!(p[1], [NaN]; color=2, label="Subsequences")
    for i in axes(out, 3)
        plot!(p[1], out[1,:,i], ones(size(out,2)).+i, color=2, label="")
        plot!(p[2], out[2,:,i], ones(size(out,2)).+i, color=2, label="")
        plot!(p[3], out[3,:,i], ones(size(out,2)).+i, color=2, label="")
    end

    display(p)
```
"""
function get_subsequences(
    x::AbstractArray,
    seq_len::Int;
    dim::Int=1, stride::Int=1,
    verbose::Int=0
    )::AbstractArray

    @assert stride >= 1 "`stride` has to be bigger than 0!"

    out = []    #init output

    #extract subsequences
    for i in axes(x,dim)[1:stride:end-seq_len+1]
        subseq_i = mapslices(xi -> xi[i:i+seq_len-1], x; dims=dim)
        push!(out, subseq_i)
    end

    #combine to array
    out = cat(out...; dims=ndims(x)+1)

    return out

end


"""
    - function to split `x` into different subsets (partitions)
    - `x` will be split along dimension `dim`

    Parameters
    ----------
        - `x`
            - `Tuple{Vararg{AbstractArray}}`, `Tuple{Vararg{DataFrame}}`
            - datsets to be split into subsets along `dim`
                - `dim` is interpreted as dimension of samples
        - `splits`
            - `Tuple{Vararg{AbstractFloat}}`, `Tuple{Vararg{Int}}`, optional
            - partition sizes of the different splits as a fraction
            - will create as many partitions as elements in `splits`
            - if `Tuple{Vararg{AbstractFloat}}`
                - each partition will contain the fraction of the initial data volume specified at that very index
                - has to sum up to `1`
                - the default is `(0.6,0.4)`
                    - will create two splits
                        - one with 60% of the input data
                        - one with 40% of the input data
            - if `Tuple{Vararg{Int}}`
                - each partition will contain exactly the specified amount of samples
                - has to sum to the total number of samples
                - defaults to `Tuple{Vararg{AbstractFloat}}`
            - the default is `(0.6,0.4)`
                - will create two splits
                    - one with 60% of the input data
                    - one with 40% of the input data
        - `shuffle`
            - `Bool`, optional
            - whether to shuffle the data before splitting
            - the default is `true`
        - `dim`
            - `Int`, optional
            - dimension along which to apply the partitioning
            - refers to the dimensions of the passed `x`
                - `dim` has to be `1` if `x` contains `DataFrame`s
            - the default is `2`
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`

    Raises
    ------
        - `AssertionError`
            - if `splits` does not sum to 1

    Returns
    -------
        - `out`
            - `Array{AbstractArray}`, `Array{DataFrames}`
            - split versions of the entries in `x`
            - will return for each `x` as many splits as passed in `splits`

    Dependencies
    ------------

    Comments
    --------

    Examples
    --------
```julia
    x1 = collect(reshape(1:(5*30), 5, 30))
    x2 = collect(reshape(1:(2*30), 2, 30))
    x = (x1, x2)
    # x = Tuple(eachslice(reshape(1:(10*5*30), 10, 5, 30); dims=1))
    println(size.(x))

    splits = split_along_dim(x, (.6,.2,.2); shuffle=true, dim=2)
    println(size.(splits))
    splits = split_along_dim(x, (20,6,4); shuffle=true, dim=2)
    println(size.(splits))
    splits = split_along_dim(x; shuffle=true, dim=2)
    println(size.(splits))

    df1 = DataFrame(x1', ["c\$i" for i in axes(x1, 1)])
    df2 = DataFrame(x2', ["col\$i" for i in axes(x2, 1)])
    splits = split_along_dim((df1,df2), (.6, .2, .2); shuffle=true)
    # splits = split_along_dim((df1,df2), (20, 6, 4); shuffle=true)
    println(size.(splits))
    println(names.(splits))
```
"""
function split_along_dim(
    x::Tuple{Vararg{AbstractArray}},
    splits::Tuple{Vararg{AbstractFloat}}=(0.6,0.4);
    shuffle::Bool=true,
    dim::Int=2,
    verbose::Int=0,
    )::Array{AbstractArray}
    
    #number of samples in each `x`
    nsamples = map(xi -> size(xi,dim), x)
    
    #check if splits sums to 1 (percentage)
    @assert sum(splits) - 1 < 1e-12 "`splits` has to sum to 1!"
    @assert allequal(nsamples) "all `x` have to have the same size along `dim` but they have the following: $(join(nsamples,", "))!"

    #get total number of samples
    ninputs  = length(nsamples)
    nsamples = nsamples[1]

    #shuffle samples if requested
    if shuffle
        ridx = Random.randperm(nsamples)
        x = [mapslices(xi->xi[ridx], x[i], dims=dim) for i in 1:ninputs]
    end

    #init output array
    out::Array = []
    for i in 1:ninputs  #iterate over all `x` (split all passed `x`)
        #init array containing splits
        x_split = Array{Any}(undef, size(splits, 1))
        
        #init starting index
        startidx::Int = 1
        for (idx, s) in enumerate(splits)   #generate splits for each passed `x`
            
            #update endidx
            endidx = convert(Int, startidx + floor(nsamples*s; digits=0) - 1)
            
            #store split
            x_split[idx] = selectdim(x[i], dim, startidx:endidx)
            # println("$idx, $startidx, $endidx, $(size(x_split[idx]))")
            
            #update startidx
            startidx = endidx + 1
        end
        
        #store array containing splits
        push!(out, x_split...)
    end

    return out
end
function split_along_dim(
    x::Tuple{Vararg{AbstractArray}},
    splits::Tuple{Vararg{Int}};
    shuffle::Bool=true,
    dim::Int=2,
    verbose::Int=0,
    )::Array{AbstractArray}
    
    #number of samples in each `x`
    nsamples = map(xi -> size(xi,dim), x)
    
    #check if splits sums to 1 (percentage)
    @assert allequal(nsamples) "all `x` have to have the same size along `dim` but they have the following: $(join(nsamples,", "))!"
    @assert sum(splits) - nsamples[1] == 0 "`splits` has to sum to `nsamples=$(nsamples[1])`!"

    #get total number of samples
    ninputs  = length(nsamples)
    nsamples = nsamples[1]

    #shuffle samples if requested
    if shuffle
        ridx = Random.randperm(nsamples)
        x = [mapslices(x->x[ridx], x[i], dims=dim) for i in 1:ninputs]
    end

    #init output array
    out::Array = []
    for i in 1:ninputs  #iterate over all `x` (split all passed `x`)
        #init array containing splits
        x_split = Array{Any}(undef, size(splits, 1))
        
        #init starting and ending index
        startidx::Int = 1
        endidx::Int = 0
        for (idx, s) in enumerate(splits)   #generate splits for each passed `x`
            
            #update endidx
            endidx += s
            
            #store split
            x_split[idx] = selectdim(x[i], dim, startidx:endidx)
            # println("$idx, $startidx, $endidx, $(size(x_split[idx]))")
            
            #update startidx
            startidx = endidx + 1
        end
        
        #store array containing splits
        push!(out, x_split...)
    end

    return out    
end
function split_along_dim(
    x::Tuple{Vararg{DataFrames.DataFrame}},
    splits::Tuple{Vararg{Int}};
    shuffle::Bool=true,
    verbose::Int=0,
    )::AbstractArray{DataFrames.DataFrame}

    nsplits = length(splits)
    #convert to processable data
    columns = names.(x)
    data = Matrix.(x)

    out = split_along_dim(
        data, splits;
        shuffle=shuffle,
        dim=1,
        verbose=verbose,
    )
    #convert back to DataFrame
    out = DataFrames.DataFrame.(out,:auto)
    
    #rename with original columns
    for s in eachindex(out)
        i = div(s-1,nsplits)+1  #current input sample
        rename!(out[s], columns[i])
    end
    
    return out
end
function split_along_dim(
    x::Tuple{Vararg{DataFrames.DataFrame}},
    splits::Tuple{Vararg{AbstractFloat}}=(0.6,0.4);
    shuffle::Bool=true,
    verbose::Int=0,
    )::AbstractArray{DataFrames.DataFrame}

    nsplits = length(splits)
    #convert to processable data
    columns = names.(x)
    data = Matrix.(x)

    out = split_along_dim(
        data, splits;
        shuffle=shuffle,
        dim=1,
        verbose=verbose,
    )
    #convert back to DataFrame
    out = DataFrames.DataFrame.(out,:auto)
    
    #rename with original columns
    for s in eachindex(out)
        i = div(s-1,nsplits)+1  #current input sample
        rename!(out[s], columns[i])
    end
    
    return out
end

end #module