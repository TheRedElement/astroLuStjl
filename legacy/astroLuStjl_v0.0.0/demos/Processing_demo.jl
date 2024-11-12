
#%%imports

include(joinpath(@__DIR__, "../src/Preprocessing.jl"))


begin #dataset splitting

    #AbstractArray (any dim possible)
    x1 = collect(reshape(1:(5*30), 5, 30))
    x2 = collect(reshape(1:(2*30), 2, 30))
    x = (x1, x2)
    # x = Tuple(eachslice(reshape(1:(10*5*30), 10, 5, 30); dims=1))

    splits = split_along_dim(x, (.6,.2,.2); shuffle=true, dim=2)
    println(size.(splits))
    splits = split_along_dim(x, (20,6,4); shuffle=true, dim=2)
    println(size.(splits))
    splits = split_along_dim(x; shuffle=true, dim=2)
    println(size.(splits))

    #DataFrames (only dim=1 possible)
    df1 = DataFrame(x1', ["c$i" for i in axes(x1, 1)])
    df2 = DataFrame(x2', ["col$i" for i in axes(x2, 1)])
    splits = split_along_dim((df1,df2), (.6, .2, .2); shuffle=true)
    # splits = split_along_dim((df1,df2), (20, 6, 4); shuffle=true)
    println(size.(splits))
    println(names.(splits))

end

begin #subsequence extraction
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
end