

#%%imports
using Revise

include(joinpath(@__DIR__, "../src/Binning.jl"))

#%%generating bins

#centered
centers = range(-3,3,20)
bins = generate_bins_centered(centers; bin_width=.9)
h = histogram(randn(500); bins=bins, label="")
vline!(bins, label="Generated Bins")
vline!(centers, label="Bin Centers")
display(h)

#based on physics
times = range(0,100,1000) .+ rand(1:1:10, 1000) #d
bins = generate_bins_phys(
    times, 25,
)
p = scatter(times, ones(size(times)), alpha=0.1, xlabel="Time")
vline!(bins)
display(p)

#%%executing data-binning
x = [
    sort(rand(-1:.01:0, 100)),
    sort(rand(0:.01:1, 100)),
    collect(range(-1,0, 101)),
]
x[1][[5,9,10]] .= NaN
y = map(x -> x.^2 .+ .05*randn(size(x)), x)

# bins = generate_bins_int.(
#     x, .1;
#     xmin=nothing, xmax=nothing,
#     verbose=0
# )
bins = generate_bins_pts.(
    x, .1;
    # x, 5;
    verbose=0
)

bng = Binning.(bins; std_func_x=nanvar)
br = fit.(bng, x, y)
res = transform.(br)
x_binned    = [res[i][1] for i in eachindex(res)]
y_binned    = [res[i][2] for i in eachindex(res)]
x_std       = [res[i][3] for i in eachindex(res)]
y_std       = [res[i][4] for i in eachindex(res)]
println(size.(x_binned))
println(size.(y_binned))

# plot!(p, binned[1]; bins=bins[1], x=x[1], y=y[1])
p1 = plot(br[1]; bins=bins[1], x=x[1], y=y[1])
p2 = plot(br[2]; bins=bins[2], x=x[2], y=y[2])
p3 = plot(br[3]; bins=bins[3], x=x[3], y=y[3])
p = plot(p1, p2, p3; layout=(1,3), size=(1200,400))
display(p)

