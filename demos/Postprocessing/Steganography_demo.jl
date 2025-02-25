
#%%imports
using Plots
using Random
using Revise

using astroLuStjl.Postprocessing.Steganography
using astroLuStjl.Styles.PlotStyleLuSt
using astroLuStjl.Styles.FormattingUtils

theme(:tre_dark)

#%%definitions
"""
    - function to generate some host image

    Parameters
    ----------
        - `xpix`
            - `Int`
            - number of pixels in x-direction
        - `ypix`
            - `Int`
            - number of pixels in y-direction

    Raises
    ------

    Returns
    -------
        - `x_host`
            - `AbstractMatrix{RGB}`
            - has shape `(xpix,ypix)`
            - elements are `RGB` channels
            - composite of different geometric functions in each channel

    Dependencies
    ------------
        - `ImageCore`

    Comments
    --------
"""
function get_host(xpix::Int, ypix::Int)::AbstractMatrix
    xy = range(-1,1,xpix) .* range(-1,1,ypix)'
    xpix, ypix = size(xy)
    x_host = zeros(xpix, ypix, 3)
    x_host[:,:,1] .+= xy.^3
    x_host[:,:,2] .+= xy.^2
    x_host[:,:,3] .+= -xy

    x_host .= (x_host .- minimum(x_host)) ./ maximum(x_host .- minimum(x_host))

    x_host = RGB.(x_host[:,:,1], x_host[:,:,2], x_host[:,:,3])
    return x_host
end

#%%demos
begin #LSBSteganography

    #setup
    x = "This is a secret message hidden in the image..."   #message to be transmitted
    println("x: $x")
    nbits = 8                   #number of bits to use for the encoding
    x_host = get_host(80,50)    #some host image
    
    #init LSBSteganography
    lsbs = Steganography.LSBSteganography(;encpix=randperm(prod(size(x_host)))[1:150])
    
    #fit to host
    lsbs = Steganography.fit(lsbs, x_host, nbits)   
    
    #encode message
    x_enc, encpix, nbits = Steganography.transform(lsbs, x)
    
    #decode message
    x_dec = Steganography.invtransform(lsbs, x_enc, encpix, nbits)
    println("x_dec: $x_dec")

    enc_mask = zeros(prod(size(x_host)))
    enc_mask[encpix] .= 1
    enc_mask = Gray.(reshape(enc_mask, size(x_host)...))

    h1 = heatmap(x_host; title="Host")
    h2 = heatmap(x_enc; title="Encoded")
    h3 = heatmap(enc_mask; title="Encoded Pixel")
    display(plot(h1, h2, h3;
        layout=(1,3),
        xlabel="Pixel", ylabel="Pixel",
        xlims=(0.5,size(x_host,2)+0.5), ylims=(0.5,size(x_host,1)+0.5),
    ))

    #`encpix` and `nbits` have to be correct to encode the message
    x_dec1 = Steganography.invtransform(lsbs, x_enc, collect(1:length(encpix)), nbits)
    x_dec2 = Steganography.invtransform(lsbs, x_enc, encpix, 16)

    println("x_dec1 (wrong `encpix``): $x_dec1")
    println("x_dec2 (wrong `nbits`): $x_dec2")

end
