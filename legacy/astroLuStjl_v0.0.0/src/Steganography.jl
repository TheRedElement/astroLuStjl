
"""
    - module to do steganography (hiding sectret information within data)

    Structs
    -------
        - `LSBSteganography`

    Functions
    ---------
        - `convert2dec()`
        - `fit()`
        - `encode()`
        - `decode()`
        - `transform()`
        - `invtransform()`
    
    Extended Functions
    ------------------ 

    Dependencies
    ------------
        - `ImageCore`
    
    Examples
    --------
        -  see [../demos/Steganography_demo.jl](../demos/Steganography_demo.jl)
"""

module Steganography

#%%imports
using ImageCore

#import for extending

#intradependencies

#%%exports
export convert2dec
export LSBSteganography
export fit
export encode
export decode
export transform
export invtransform


#%%definitions
#######################################
#helper functions
"""
    - function to convert an array of coefficients to base 10 given some `base`

    Parameters
    ----------
        - `x`
            - `AbstractVector`
            - contains coefficients in some basis `base`
            - for the default (`base=2`) use binary bits
        - `base`
            - `Int`, optional
            - base to use for the conversion
            - the default is `2`
                - binary number

    Raises
    ------

    Returns
    -------
        - `x_dec`
            - `Int`
            - `x` in base 10

    Dependencies
    ------------

    Comments
    --------
"""
function convert2dec(
    x::AbstractVector;
    base::Int=2
    )::Int

    dec = sum(x .* (base .^(eachindex(x).-1)))
    return dec
end

#######################################
#Steganography based on LSB using an image
"""
    - struct to apply steganography based on modifying the least significant bit (LSB) of some host image

    Fields
    ------
        - `encpix`
            - `Vector`, optional
            - indices of pixels containing the hidden information
            - the default is `nothing`
                - will be set during fitting
                    - uses first `npix` pixels that suffice to encode the requested information
        - `x_host_bs`
            - `AbstractArray`
            - bitstream version of the host image `x_host`
            - has size `(xpix,ypix,nchannels,nbits)`
            - the default is `nothing`
                - set during fitting
        - `state`
            - `Symbol`, optional
            - can be one of
            - `:init`
                - the model has been initialized
            - `:fitted`
                - the model has been fitted
            - the default is `:init`
    
    Methods
    -------
        - `fit()`
        - `encode()`
        - `decode()`
        - `transform()`
        - `invtransform()`

    Comments
    --------

"""
struct LSBSteganography
        encpix::Union{Vector,Nothing}
        x_host_bs::Union{AbstractArray,Nothing}
        state::Symbol

    #inner constructor (to allow default values)
    function LSBSteganography(
        ;
        encpix::Union{Vector,Nothing}=nothing,
        x_host_bs::Union{AbstractArray,Nothing}=nothing,
        state::Symbol=:init,
        )

        @assert state in [:init,:fitted] ArgumentError("`state` has to be one of `:init`, `:fitted` but is `$state`!")

        new(encpix,x_host_bs,state)
    end    
end

"""
    - method to fit `LSBSteganography` transformer
    
    Parameters
    ----------
        - `lsbs`
            - `LSBSteganography`
            - instance of `LSBSteganography` containing the model initialization
        - `x_host`
            - `AbstractMatrix{RGB{Real}}`
            - host image
            - elements contain `RGB`-arrays
            - will serve as canvas to hide a message
        - `nbits`
            - `Int`
            - number of bits to use for the encoding
            - one of the keys to decoding the data after encoding
            - for encoding standard ASCII characters `nbits=8` is sufficient
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`

    Raises
    ------

    Returns
    -------
        - `lsbs`
            - `LSBSteganography`
            - fitted instance of `LSBSteganography`

    Comments
    --------
"""
function fit(
    lsbs::LSBSteganography,
    x_host::AbstractMatrix{RGB{T}},
    nbits::Int;
    verbose::Int=0
    )::LSBSteganography where {T <: Real}
    
    #convert host to bitstream
    x_host_bs = Int.(round.(permutedims(channelview(copy(x_host)), (2,3,1)) .* 255))   #`(xpix,ypix,nchannels)`
    x_host_bs = digits.(x_host_bs; base=2, pad=nbits)
    x_host_bs = permutedims(stack(x_host_bs), (2,3,4,1))                                #`(xpix,ypix,nchannels,nbits)`
    xpix, ypix, nchannels, nbits = size(x_host_bs)

    #get indices of encoding-pixels
    encpix = isnothing(lsbs.encpix) ? collect(range(1,xpix*ypix)) : lsbs.encpix

    return LSBSteganography(;encpix=encpix, x_host_bs=x_host_bs, state=:fitted)
end

"""
    - function to encode some message `x` into the host `x_host`

    Parameters
    ----------
        - `lsbs`
            - `LSBSteganography`
            - fitted instance of `LSBSteganography`
        - `x`
            - `String`
            - message to be encrypted into `x_host`
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`
    
    Raises
    ------
        - `AssertionError`
            - if the model has not been fit yet
            - if not enough pixel indices have been provided to encode the entire message `x`

    Returns
    -------
        - `x_enc`
            - `AbstractMatrix{RGB{Real}}`
            - encoded version of `x`
            - the host image with `x` encoded into `encpix` using `nbits` bits
        - `encpix`
            - `Vector`
            - indices of pixels containing the hidden information
            - one of the keys to decoding `x_enc` and reconstructing `x`
        - `nbits`
            - `Int`
            - number of bits used for the encoding
            - one of the keys to decoding `x_enc` and reconstructing `x`

    Comments
    --------
"""
function encode(
    lsbs::LSBSteganography,
    x::String;
    verbose::Int=0,
    )::Tuple{AbstractMatrix,Vector,Int}

    @assert lsbs.state == :fitted "`lsbs` has not been fitted yet. make sure to call `fit(lsbs,...)` before transforming"

    #get host-specifications
    xpix, ypix, nchannels, nbits = size(lsbs.x_host_bs)
    
    #init encoded version (convert to pixel domain)
    x_enc = reshape(lsbs.x_host_bs, :, nchannels, nbits)   

    #encode input
    x = x * ("_"^(nchannels-length(x)%nchannels))   #pad to make length divisible by 3 (3 channels)
    x = only.(split(x, ""))                         #extract characters
    x = Int.(x)                                     #convert to ascii decimal
    x = digits.(x; base=2, pad=nbits)               #convert to bits (8 bits for values 0:255 - rgb - needed)
    x = mapreduce(permutedims, vcat, x)             #convert to matrix
    x = reshape(x, :, nchannels, 1)                 #`nchannels` (typically 3 - rbg) bits can be encoded per pixel #`(npixels,nchannels,nbits)`
    npix, nchannels_, nbits_ = size(x)

    #get pixel indices to use for encoding (first `npix`)
    @assert length(lsbs.encpix) >= npix "`length(lsbs.encpix)` has to be greater or equal to `npix=$npix`!"
    encpix = lsbs.encpix[1:npix]

    #modify least significant bits (LSB) of each channel in all `encpix` pixels
    x_enc[encpix,:,1] .= x   #encode in LSB
    
    #convert back to image (includes encoded `String`)
    x_enc = reshape(x_enc, xpix, ypix, nchannels, nbits)
    x_enc = dropdims(mapslices(convert2dec, x_enc; dims=4); dims=4) ./ 255
    x_enc = RGB.(x_enc[:,:,1],x_enc[:,:,2],x_enc[:,:,3])

    return x_enc, encpix, nbits
end

"""
    - method to decode `x_enc` and retrieve the hidden message

    Parameters
    ----------
        - `lsbs`
            - `LSBSteganography`
            - not used, but needed for function signature consistency
        - `x_enc`
            - `AbstractMatrix{RGB}`
            - host-image containining message encoded with `nbits` bits in pixels `encpix`
        - `encpix`
            - `Vector`
            - indices of pixels containing the hidden information
            - one of the keys to decoding `x_enc` and reconstructing `x`
        - `nbits`
            - `Int`
            - number of bits used for the encoding
            - one of the keys to decoding `x_enc` and reconstructing `x`
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`

    Raises
    ------

    Returns
    -------
        - `x_dec`
            - `String`
            - the decoded hidden message 

    Comments
    --------
"""
function decode(
    lsbs::LSBSteganography,
    x_enc::AbstractMatrix,
    encpix::Vector, nbits::Int;
    verbose::Int=verbose
    )::String

    #decode
    x_dec = Int.(round.(permutedims(channelview(x_enc), (2,3,1)) .* 255))
    x_dec = digits.(x_dec; base=2, pad=nbits)
    x_dec = permutedims(stack(x_dec), (2,3,4,1))        #`(xpix,ypix,nchannels,nbits)`
    xpix, ypix, nchannels, nbits_ = size(x_dec)
    x_dec = reshape(x_dec, :, nchannels, nbits)         #`(xpix*ypix,nchannels,nbits)`

    x_dec = x_dec[encpix,:,1]                             #extract LSBs of pixels containing the message
    x_dec = reshape(x_dec, :, nbits)                        #combine to the number of bits

    #convert to base 10
    x_dec = dropdims(mapslices(x -> convert2dec(x; base=2), x_dec; dims=2); dims=2)
    
    #lookup in ASCII table and combine to entire message
    x_dec = join(Char.(x_dec), "")

    return x_dec
end

"""
    - function to encode some message `x` into the host `x_host`
    - equivalent to `encode()`

    Parameters
    ----------
        - `lsbs`
            - `LSBSteganography`
            - fitted instance of `LSBSteganography`
        - `x`
            - `String`
            - message to be encrypted into `x_host`
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`
    
    Raises
    ------
        - `AssertionError`
            - if the model has not been fit yet
            - if not enough pixel indices have been provided to encode the entire message `x`

    Returns
    -------
        - `x_enc`
            - `AbstractMatrix{RGB{Real}}`
            - encoded version of `x`
            - the host image with `x` encoded into `encpix` using `nbits` bits
        - `encpix`
            - `Vector`
            - indices of pixels containing the hidden information
            - one of the keys to decoding `x_enc` and reconstructing `x`
        - `nbits`
            - `Int`
            - number of bits used for the encoding
            - one of the keys to decoding `x_enc` and reconstructing `x`

    Comments
    --------
"""
function transform(
    lsbs::LSBSteganography,
    x::String;
    verbose::Int=0
    )::Tuple{AbstractMatrix,Vector,Int}

    x_enc, encpix, nbits = encode(lsbs, x; verbose=verbose)
    return x_enc, encpix, nbits
end

"""
    - method to decode `x_enc` and retrieve the hidden message
    - equivalent to `encode()`

    Parameters
    ----------
        - `lsbs`
            - `LSBSteganography`
            - not used, but needed for function signature consistency
        - `x_enc`
            - `AbstractMatrix{RGB}`
            - host-image containining message encoded with `nbits` bits in pixels `encpix`
        - `encpix`
            - `Vector`
            - indices of pixels containing the hidden information
            - one of the keys to decoding `x_enc` and reconstructing `x`
        - `nbits`
            - `Int`
            - number of bits used for the encoding
            - one of the keys to decoding `x_enc` and reconstructing `x`
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`

    Raises
    ------

    Returns
    -------
        - `x_dec`
            - `String`
            - the decoded hidden message 

    Comments
    --------
"""
function invtransform(
    lsbs::LSBSteganography,
    x_enc::AbstractMatrix,
    encpix::Vector, nbits::Int;
    verbose::Int=0,
    )::String

    x_dec = decode(lsbs, x_enc, encpix, nbits; verbose=verbose)
    return x_dec
    
end

end #module

