
#%%imports
using LaTeXStrings
using Plots
using Revise

include(joinpath(@__DIR__, "../src/Bezier3Interp.jl"))
using .Bezier3Interp

#%%cubic bezier interpolation
#instantiate transformer
b3i = Bezier3Interp.Bz3Interp()
println(b3i)

#data to be interpolated
x = rand(0:.1:20, (3,7))

#fit
b3i = Bezier3Interp.fit(b3i, x; verbose=0)
println(b3i)

#generate interpolated paths
x_interp = Bezier3Interp.transform(b3i, 6*20)


#plot result
p1 = Bezier3Interp.plot(b3i; x=x) #2d projection

## custom 3d visualzation
p2 = scatter3d(x[1,:], x[2,:], x[3,:]; xlabel=L"$x$", ylabel=L"$y$", zlabel=L"$z$", label="", title="3d-View")
plot!(p2, x_interp[1,:], x_interp[2,:], x_interp[3,:], label="")

p = plot(p1, p2)

display(p)

