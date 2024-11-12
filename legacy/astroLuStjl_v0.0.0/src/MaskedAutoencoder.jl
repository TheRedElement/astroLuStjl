
#NOTE: spatial position encoding in `PatchProjection()` not improving upon plain version

"""
    - module defining structs and function to build a Masked Autoencoder (MAE)
	- following
		- [He et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021arXiv211106377H/abstract)
		- [Dosovitskiy et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020arXiv201011929D/abstract)

    Structs
    -------

    Functions
    ---------

    Dependencies
    ------------
        - `DataFrames`
        - `DataFramesMeta`
		- `Flux`
        - `JLD2`
		- `FormattingUtils.jl`

"""

#%%imports
using DataFrames
using DataFramesMeta
using Flux
using JLD2

include("./FormattingUtils.jl")
using .FormattingUtils: printlog

#%%definitions


#Patch Projection
#----------------
"""
	- custom layer implementing projection of each series of patches onto vector

	Fields
	------
		- `proj_layer` (trainable)
			- `Flux.Dense`
			- dense layer for the (linear) projection
		- `size_in` (non-trainable)
			- `Tuple{Int,Int}`
			- input size
			- dimensions of each patch
				- i.e. size = `(X,Y)`
		- `size_emb` (non-trainable)
			- `Int`
			- size each patch will be projected onto
		- `nparams`
			- `Int`
			- number of learnable parameters
	
	Methods
	-------
		- `PatchProjection()`


	Dependencies
	------------
		- `Flux`

	Comments
	--------
		- abbreviations for commenting
			- X ... patch dimension in x
			- Y ... patch dimension in y
			- S ... sequence lenght
			- B ... batch size
			- E ... embedded size

"""
struct PatchProjection
	proj_layer::Flux.Dense
	size_in::Tuple{Int,Int}
	size_emb::Int
	nparams::Int
end
"""
	- function to initialize a `PatchProjection` layer
	
	Parameters
	----------
		- `size_patch`
			- `Tuple{Int,Int}`
			- size of each patch in the dataset
		- `size_emb`
			- `Int`
			- size each patch shall be projcted onto

	Raises
	------

	Returns
	-------
		- `PatchProjection`
			- initialized instance of `PatchProjection`

	Comments
	--------
"""
function PatchProjection(
	size_patch::Tuple{Int,Int},
	size_emb::Int
	)::PatchProjection
	X, Y = size_patch
	dense_in = X*Y	#input size = flattened images
	
	#TODO: add Conv2d-layer

	#projection
	dense = Flux.Dense(dense_in, size_emb)
	nparams = sum(length.(Flux.params(dense)))

	return PatchProjection(dense, size_patch, size_emb, nparams)
end
"""
	- forward pass of `PatchProjection`

	Parameters
	----------
		- `x`
			- `AbstractArray`
			- input data to be propagated through the layer
		- `train`
			- `Bool`, optional
			- whether the model is in training mode or not
			- important for gradient computation, as `Zygote` cannot deal with user defined functions in `Flux.gradient()`
			- setting to `true` will remove all custom function calls
			- the default is `false`
		- `verbose`
			- `Int`, optional
			- verbosity level
			- the default is `0`
	
	Raises
	------

	Returns
	-------
		- `x_enc`
			- `AbstractArray`
			- encoded version of `x`

	Comments
	--------
"""
function(pp::PatchProjection)(
	x::AbstractArray{Float32};
	train::Bool=false,
	verbose::Int=0,
	)::AbstractArray{Float32}
	X, Y, S, B = size(x)

	#= Draft for Conv
		c_filter 	= (5,5)
		stride 		= 1
		pad			= 0
		d_conv 		= 4
		d_ff		= 64
		
		X_c = Int32((X+2*pad-c_filter[1])/stride + 1)
		Y_c = Int32((Y+2*pad-c_filter[2])/stride + 1)

		println(X_c, Y_c)

		c 	= Flux.Conv(c_filter, 1 => d_conv; stride=stride, pad=pad)
		ff 	= Flux.Chain(
			Flux.Dense(X_c*Y_c => d_ff),
			Flux.Dense(d_ff => X_c*Y_c)
		)
		ct	= Flux.ConvTranspose(c_filter, d_conv => 1; stride=stride, pad=pad)


		x_c = reshape(x, X, Y, 1, S*B)	#add channel dimension
		x_c = c(x_c)					#propagate through Conv
		x_c = reshape(x_c, X_c*Y_c, d_conv*S*B)
		x_c = ff(x_c)
		x_c = reshape(x_c, X_c, Y_c, d_conv, S*B)
		x_c = ct(x_c)
		println(size(x_c))
	=#

	x_flat = reshape(x, X*Y, S*B)	#(X*Y,S*B)
	x_enc = pp.proj_layer(x_flat)	#apply Dense layer on all patches (project)
	x_enc = reshape(x_enc, pp.size_emb, S, B)	#(E,S,B)

	#summary
	if ~train && verbose > 0
		printlog(
			"$(size(x)) -> $(size(x_enc)); $(pp.nparams) params\n";
			context="PatchProjection()",
			type=:INFO,
			level=2,
			verbose=verbose-1,
		)
	end

	return x_enc
end
#register layer to flux (only `proj_layer` is trainable)
Flux.@functor PatchProjection (proj_layer,)

#Position Encoding
#-----------------
"""
	- custom layer implementing learnable position encoding

	Fields
	------
		- `w_enc` (trainable)
			- `AbstractArray{Float32}`
			- encoding matrix
			- has size `(d_model,n_samples)`
		- `nparams`
			- `Int`
			- number of learnable parameters
	Methods
	-------
		- `LearnablePositionEncoding()`

	Dependencies
	------------
		- `Flux`

	Comments
	--------
"""
struct LearnablePositionEncoding
	w_enc::AbstractArray{Float32}
	nparams::Int
end
"""
	- function to initialize `LearnablePositionEncoding` layer

	Parameters
	----------
		- `d_model`
			- `Int`
			- dimension of the (transformer) model
		- `max_len`
			- `Int`
			- maximum number of samples per batch

	Raises
	------

	Returns
	-------
		- `LearnablePositionEncoding`
			- initialized instance of `LearnablePositionEncoding`

	Comments
	--------

"""
function LearnablePositionEncoding(
	d_model::Int, max_len::Int,
	)::LearnablePositionEncoding
	
	w_enc = Flux.glorot_uniform(d_model, max_len)
	nparams = sum(length.(Flux.params(w_enc)))
	
	return LearnablePositionEncoding(w_enc, nparams)
end
"""
	- forward pass of `LearnablePositionEncoding`
	
	Parameters
	----------
		- `x`
			- `AbstractArray`
			- input data to be propagated through the layer
		- `train`
			- `Bool`, optional
			- whether the model is in training mode or not
			- important for gradient computation, as `Zygote` cannot deal with user defined functions in `Flux.gradient()`
			- setting to `true` will remove all custom function calls
			- the default is `false`
		- `verbose`
			- `Int`, optional
			- verbosity level
			- the default is `0`
	
	Raises
	------

	Returns
	-------
		- `x_enc`
			- `AbstractArray`
			- encoded version of `x`

	Comments
	--------

"""
function (lpe::LearnablePositionEncoding)(
	x::AbstractArray{Float32};
	train::Bool=false,
	verbose::Int=0,
	)::AbstractArray{Float32}

	x_enc = x .+ lpe.w_enc

	#summary
	if ~train && verbose > 0
		printlog(
			"$(size(x)) -> $(size(x_enc)); $(lpe.nparams) params\n";
			context="LearnablePositionEncoding()",
			type=:INFO,
			level=2,
			verbose=verbose-1,
		)
	end

	return x_enc

end
Flux.@functor LearnablePositionEncoding (w_enc,)

#Transformer Encoder Block
#-------------------------
"""
	- custom layer implementing a transformer encoder block

	Fields
	------
		- `mha` (trainable)
			- `Flux.MultiHeadAttention`
			- layer to exectue multi head attention
		- `ln1` (trainable)
			- `Flux.LayerNorm`
			- layer to normalize the output of `mha`
		- `ff` (trainable)
			- `Flux.Chain`
			- feedforward block after `mha`
		- `ln2` (trainable)
			- `Flux.LayerNorm`
			- layer to normalize outputs of `ff`
		- `nparams`
			- `Int`
			- number of learnable parameters

	Methods
	-------
		- `TransformerEncoderBlock()`

	Dependencies
	------------
		- `Flux`

	Comments
	--------
"""
struct TransformerEncoderBlock
	mha::Flux.MultiHeadAttention
	ln1::Flux.LayerNorm
	ff::Flux.Chain
	ln2::Flux.LayerNorm
	nparams::Int
end
"""
	- function to initialize a `TransformerEncoderBlock` layer

	Parameters
	----------
		- `d_model`
			- `Int`
			- model dimension
			- i.e. hidden dimension of the model
		- `d_head`
			- `Int`
			- head dimension
			- total number of neurons distributed across attention heads
			- has to be divisible by `n_heads`
		- `n_heads`
			- `Int`
			- number of attention heads to use in the model
		- `d_ff`
			- `Int`
			- dimension of the feedforward block following the multi head attention
		- `act_ff`
			- `Function`, optional
			- activation function to use in the feedforward block
			- the default is `Flux.relu`
		- `mha_dropout`
			- `Float32`, optional
			- dropout in the multi-head attention
			- the default is `0.`			

	Raises
	------

	Returns
	-------
		- `TransformerEncoderBlock`
			- initialized instance of `TransformerEncoderBlock`

	Comments
	--------
"""
function TransformerEncoderBlock(
	d_model::Int,
	d_head::Int, n_heads::Int,
	d_ff::Int;
	act_ff::Function=Flux.relu,
	mha_dropout::AbstractFloat=0.,
	)::TransformerEncoderBlock

	mha = Flux.MultiHeadAttention(d_model => d_head => d_model; nheads=n_heads, bias=false, dropout_prob=mha_dropout)
	ln1 = Flux.LayerNorm(d_model)
	ff = Flux.Chain(
		Flux.Dense(d_model, d_ff, act_ff),
		Flux.Dense(d_ff, d_model)
	)
	ln2 = Flux.LayerNorm(d_model)
	nparams = sum(length.(Flux.params(mha, ln1, ff, ln2)))


	return TransformerEncoderBlock(mha, ln1, ff, ln2, nparams)
end
"""
	- forward pass of `TransformerEncoderBlock`
	
	Parameters
	----------
		- `x`
			- `AbstractArray`
			- input data to be propagated through the layer
		- `mha_mask`
			- `BitMatrix`, `Nothing`, optional
			- multi-head attention mask in `Flux.MultiHeadAttention()`
			- the default is `nothing`
				- no mask applied
		- `train`
			- `Bool`, optional
			- whether the model is in training mode or not
			- important for gradient computation, as `Zygote` cannot deal with user defined functions in `Flux.gradient()`
			- setting to `true` will remove all custom function calls
			- the default is `false`
		- `verbose`
			- `Int`, optional
			- verbosity level
			- the default is `0`
	
	Raises
	------

	Returns
	-------
		- `x_ln2`
			- `AbstractArray`
			- `x` after propagating through the `TransformerEncoderBlock` block
		- `attn`
			- `AbstractArray`
			- self-attention maps for each sample

	Comments
	--------

"""
function (tfeb::TransformerEncoderBlock)(
	x::AbstractArray{Float32},
	mha_mask::Union{BitMatrix,Nothing}=nothing;
	train::Bool=false,
	verbose::Int=0,
	)::Tuple{AbstractArray,AbstractArray}

	x_mha, attn = tfeb.mha(x, x, x; mask=mha_mask)
	x_ln1 = tfeb.ln1(x .+ x_mha)
	x_ff = tfeb.ff(x_ln1)
	x_ln2 = tfeb.ln2(x_ln1 .+ x_ff)

	#summary
	if ~train && verbose > 0
		#substructures
		nparams_ln2 = sum(length.(Flux.params(tfeb.ln2)))
		printlog(
			"$(size(x_ff)) -> $(size(x_ln2)); $(nparams_ln2) params\n";
			context="LayerNorm()",
			type=:INFO,
			level=3,
			verbose=verbose-2,
		)
		nparams_ff = sum(length.(Flux.params(tfeb.ff)))
		printlog(
			"$(size(x_ln1)) -> $(size(x_ff)); $(nparams_ff) params\n";
			context="Dense()",
			type=:INFO,
			level=3,
			verbose=verbose-2,
		)
		nparams_ln1 = sum(length.(Flux.params(tfeb.ln1)))
		printlog(
			"$(size(x_mha)) -> $(size(x_ln1)); $(nparams_ln1) params\n";
			context="LayerNorm()",
			type=:INFO,
			level=3,
			verbose=verbose-2,
		)
		nparams_mha = sum(length.(Flux.params(tfeb.mha)))
		printlog(
			"$(size(x)) -> $(size(x_mha))[$(size(attn))]; $(nparams_mha) params\n";
			context="MultiHeadSelfAttention()",
			type=:INFO,
			level=3,
			verbose=verbose-2,
		)

		#superstructure
		printlog(
			"$(size(x)) -> $(size(x_ln2))[$(size(attn))]; $(tfeb.nparams) params\n";
			context="TransformerEncoderBlock()",
			type=:INFO,
			level=2,
			verbose=verbose-1,
		)		
	end


	return x_ln2, attn
end
Flux.@functor TransformerEncoderBlock

#Masked Autoencoder Encoder Block
#--------------------------------
"""
	- custom layer implementing the encoder part of a Masked Autoencoder

	Fields
	------
		- `pp` (trainable)
			- `PatchProjection`
			- layer to project patches onto vectors
		- `lpe` (trainable)
			- `LearnablePositionEncoding`
			- layer to add learnable position encodings
		- `tfeb` (trainable)
			- `AbstractVector{TransformerEncoderBlock}`
			- `n_blocks` instances of `TransformerEncoderBlock`
			- the number of `TransformerEncoderBlock` the model shall have
		- `n_heads` (non-trainable)
			- `Int`
			- number of heads in all of the `TransformerEncoderBlock`
			- only needed for size-casting
		- `seq_len` (non-trainable)
			- `Int`
			- sequence length of the input to the `MAEEncoder`
			- only for convenence (later reference)
		- `nparams`
			- `Int`
			- number of learnable parameters

	Methods
	-------
		- `MAEEncoder()`

	Dependencies
	------------
		- `Flux`

	Comments
	--------
"""
struct MAEEncoder
	pp::PatchProjection
	lpe::LearnablePositionEncoding	
	tfeb::AbstractVector{TransformerEncoderBlock}
	n_heads::Int	#only for formatting in forward pass
	seq_len::Int	#only for convenience (later reference)
	nparams::Int
end
"""
	- function to initialize a `MAEEncoder` layer

	Parameters
	----------
		- `size_patch`
			- `Tuple{Int,Int}`
			- dimensions of the patches in the training set
			- has to have form `(X,Y)`
		- `seq_len`
			- `Int`
			- length of the individual sequences		
		- `d_model`
			- `Int`
			- model dimension
			- i.e. hidden dimension of the model
			- applied to each of `n_blocks` `TransformerEncoderBlock`
		- `n_blocks`
			- `Int`
			- number of `TransformerEncoderBlock` the `MAEEncoder` shall have
		- `d_head`
			- `Int`
			- head dimension
			- total number of neurons distributed across attention heads in each `TransformerEncoderBlock`
			- has to be divisible by `n_heads`
			- applied to each of `n_blocks` `TransformerEncoderBlock`
		- `n_heads`
			- `Int`
			- number of attention heads to use in each `TransformerEncoderBlock`
			- applied to each of `n_blocks` `TransformerEncoderBlock`
		- `d_ff`
			- `Int`
			- dimension of the feedforward block following the multi head attention in each `TransformerEncoderBlock`
			- applied to each of `n_blocks` `TransformerEncoderBlock`
		- `act_ff`
			- `Function`, optional
			- activation function to use in the feedforward block in each `TransformerEncoderBlock`
			- applied to each of `n_blocks` `TransformerEncoderBlock`
			- the default is `Flux.relu`
		- `mha_dropout`
			- `Float32`, optional
			- dropout in the multi-head attention in each `TransformerEncoderBlock`
			- applied to each of `n_blocks` `TransformerEncoderBlock`
			- the default is `0.`

	Raises
	------

	Returns
	-------
		- `MAEEncoder`
			- initialized instance of `MAEEncoder`

	Comments
	--------	
"""
function MAEEncoder(
	size_patch::Tuple{Int,Int}, seq_len::Int,
	d_model::Int,
	n_blocks::Int,
	d_head::Int, n_heads::Int,
	d_ff::Int;
	act_ff::Function=Flux.relu,
	mha_dropout::AbstractFloat=0.,
	)
	pp = PatchProjection(size_patch, d_model)
	lpe = LearnablePositionEncoding(d_model, seq_len)
	tfeb = [
		TransformerEncoderBlock(
			d_model,
			d_head, n_heads,
			d_ff,
			act_ff=act_ff,
			mha_dropout=mha_dropout,
		)
	for _ in 1:n_blocks]
	nparams = sum(length.(Flux.params(pp, lpe, tfeb...)))

	return MAEEncoder(pp, lpe, tfeb, n_heads, seq_len, nparams)
end
"""
	- forward pass of `MAEEncoder`
	
	Parameters
	----------
		- `x`
			- `AbstractArray`
			- input data to be propagated through the layer
		- `mha_mask`
			- `BitMatrix`, `Nothing`, optional
			- multi-head attention mask in `Flux.MultiHeadAttention()` of each `TransformerEncoderBlock`
			- the default is `nothing`
				- no mask applied
		- `train`
			- `Bool`, optional
			- whether the model is in training mode or not
			- important for gradient computation, as `Zygote` cannot deal with user defined functions in `Flux.gradient()`
			- setting to `true` will remove all custom function calls
			- the default is `false`
		- `verbose`
			- `Int`, optional
			- verbosity level
			- the default is `0`
	
	Raises
	------

	Returns
	-------
		- `x_maee`
			- `AbstractArray{Float32,3}`
			- `x` after propagating through the `MAEEncoder`
		- `attns`
			- `AbstractArray{Float32,5}`
			- self-attention maps of each `TransformerEncoderBlock`
				- each element contains the self-attention maps for each sample and each head
			- has size `(seq_len,seq_len,n_heads,n_samples,n_blocks)`

	Comments
	--------
"""
function (maee::MAEEncoder)(
	x::AbstractArray{Float32},
	mha_mask::Union{BitMatrix,Nothing}=nothing;
	train::Bool=false,
	verbose::Int=0,
	)::Tuple{AbstractArray{Float32,3},AbstractArray{Float32,5}}

	X, Y, seq_len, batch_size = size(x)
	n_blocks = size(maee.tfeb,1)
	n_heads = maee.n_heads

	#projection and embedding
	x_maee	= maee.pp(x; 		train=train, verbose=verbose)
	x_maee	= maee.lpe(x_maee, 	train=train, verbose=verbose)
	
	#propagation through all `TransformerEncoderBlock`
	attns = Flux.Zygote.Buffer(Array{Float32,5}(undef, seq_len,seq_len,n_heads,batch_size,n_blocks))	#write to buffer to allow mutable operation
	for i in eachindex(maee.tfeb)
		#pass through `TransformerEncoderBlock` `i`
		x_maee, attn = maee.tfeb[i](x_maee, mha_mask; train=train, verbose=verbose)
		
		#store attention maps
		attns[:,:,:,:,i] = attn
	end

	#summary
	if ~train && verbose > 0
		printlog(
			"$(size(x)) -> $(size(x_maee)); $(maee.nparams) params\n";
			context="MAEEncoder()",
			type=:INFO,
			level=1,
			verbose=verbose,
		)			
	end

	return x_maee, copy(attns)	#copy makes object immutable again
end
Flux.@functor MAEEncoder (pp, lpe, tfeb,)


#Transformer Decoder Block
#-------------------------
"""
	- custom layer implementing a tranformer decoder block

	Fields
	------
		- `mha` (trainable)
			- `Flux.MultiHeadAttention`
			- layer to exectue multi head attention
		- `ln1` (trainable)
			- `Flux.LayerNorm`
			- layer to normalize the output of `mha`
		- `mhca` (trainable)
			- `Flux.MultiHeadAttention`
			- layer to exectue multi head cross attention
		- `ln2` (trainable)
			- `Flux.LayerNorm`
			- layer to normalize the output of `mhca`
		- `ff` (trainable)
			- `Flux.Chain`
			- feedforward block after `mhca`
		- `ln3` (trainable)
			- `Flux.LayerNorm`
			- layer to normalize outputs of `ff`
		- `nparams`
			- `Int`
			- number of learnable parameters			

	Methods
	-------
		- `TransformerDecoderBlock()`

	Dependencies
	------------
		- `Flux`

	Comments
	--------
"""
struct TransformerDecoderBlock
	mha::Flux.MultiHeadAttention
	ln1::Flux.LayerNorm
	mhca::Flux.MultiHeadAttention
	ln2::Flux.LayerNorm
	ff::Flux.Chain
	ln3::Flux.LayerNorm
	nparams::Int
end
"""
	- function to initialize a `TransformerDecoderBlock` layer

	Parameters
	----------
		- `d_model`
			- `Int`
			- model dimension
			- i.e. hidden dimension of the model
		- `d_head`
			- `Int`
			- head dimension
			- total number of neurons distributed across attention heads
			- has to be divisible by `n_heads`
		- `n_heads`
			- `Int`
			- number of attention heads to use in the model
		- `d_ff`
			- `Int`
			- dimension of the feedforward block following the multi head attention
		- `act_ff`
			- `Function`, optional
			- activation function to use in the feedforward block
			- the default is `Flux.relu`
		- `mha_dropout`
			- `Float32`, optional
			- dropout in the multi-head attention
			- the default is `0.`
		- `mhca_dropout`
			- `Float32`, optional
			- dropout in the multi-head cross-attention
			- the default is `0.`

	Raises
	------

	Returns
	-------
		- `TransformerDecoderBlock`
			- initialized instance of `TransformerDecoderBlock`

	Comments
	--------
"""
function TransformerDecoderBlock(
	d_model::Int,
	d_head::Int, n_heads::Int,
	d_ff::Int;
	act_ff::Function=Flux.relu,
	mha_dropout::AbstractFloat=0.,
	mhca_dropout::AbstractFloat=0.,
	)::TransformerDecoderBlock

	mha = Flux.MultiHeadAttention(d_model => d_head => d_model; nheads=n_heads, bias=false, dropout_prob=mha_dropout)
	ln1 = Flux.LayerNorm(d_model)
	mhca = Flux.MultiHeadAttention(d_model => d_head => d_model; nheads=n_heads, bias=false, dropout_prob=mhca_dropout)
	ln2 = Flux.LayerNorm(d_model)
	ff = Flux.Chain(
		Flux.Dense(d_model, d_ff, act_ff),
		Flux.Dense(d_ff, d_model)
	)
	ln3 = Flux.LayerNorm(d_model)
	nparams = sum(length.(Flux.params(mha, ln1, mhca, ln2, ff, ln3)))

	return TransformerDecoderBlock(mha, ln1, mhca, ln2, ff, ln3, nparams)
end
"""
	- forward pass of `TransformerDecoderBlock`
	
	Parameters
	----------
		- `x`
			- `AbstractArray{Float32}`
			- input data to be propagated through the layer
		- `x_enc`
			- `AbstractArray{Float32}`
			- output of a transformer encoder block (`TransformerEncoderBlock`)
		- `mha_mask`
			- `BitMatrix`, `Nothing`, optional
			- multi-head attention mask in `Flux.MultiHeadAttention()`
			- the default is `nothing`
				- no mask applied
		- `mhca_mask_dec`
			- `BitMatrix`, `Nothing`, optional
			- multi-head cross-attention mask in `Flux.MultiHeadAttention()`
			- the default is `nothing`
				- no mask applied
		- `train`
			- `Bool`, optional
			- whether the model is in training mode or not
			- important for gradient computation, as `Zygote` cannot deal with user defined functions in `Flux.gradient()`
			- setting to `true` will remove all custom function calls
			- the default is `false`
		- `verbose`
			- `Int`, optional
			- verbosity level
			- the default is `0`
	
	Raises
	------

	Returns
	-------
		- `x_ln3`
			- `AbstractArray`
			- `x` after propagating through the `TransformerDecoderBlock` block
		- `attn`
			- `AbstractArray`
			- self-attention maps for each sample
		- `cattn`
			- `AbstractArray`
			- cross-attention maps for each sample

	Comments
	--------

"""
function (tfdb::TransformerDecoderBlock)(
	x::AbstractArray{Float32}, x_enc::AbstractArray{Float32},
	mha_mask::Union{BitMatrix,Nothing}=nothing, mhca_mask::Union{BitMatrix,Nothing}=nothing;
	train::Bool=false,
	verbose::Int=0,
	)::Tuple{AbstractArray,AbstractArray,AbstractArray}

	x_mha, attn = tfdb.mha(x, x, x; mask=mha_mask)
	x_ln1 = tfdb.ln1(x .+ x_mha)
	x_mhca, cattn = tfdb.mhca(x_ln1, x_enc, x_enc; mask=mhca_mask)
	x_ln2 = tfdb.ln2(x_ln1 .+ x_mhca)
	x_ff = tfdb.ff(x_ln2)
	x_ln3 = tfdb.ln2(x_ln2 .+ x_ff)

	#summary
	if ~train && verbose > 0

		#substructures
		nparams_ln3 = sum(length.(Flux.params(tfdb.ln3)))
		printlog(
			"$(size(x_ff)) -> $(size(x_ln3)); $(nparams_ln3) params\n";
			context="LayerNorm()",
			type=:INFO,
			level=3,
			verbose=verbose-2,
		)
		nparams_ff = sum(length.(Flux.params(tfdb.ff)))
		printlog(
			"$(size(x_ln2)) -> $(size(x_ff)); $(nparams_ff) params\n";
			context="Dense()",
			type=:INFO,
			level=3,
			verbose=verbose-2,
		)
		nparams_ln2 = sum(length.(Flux.params(tfdb.ln2)))
		printlog(
			"$(size(x_mhca)) -> $(size(x_ln2)); $(nparams_ln2) params\n";
			context="LayerNorm()",
			type=:INFO,
			level=3,
			verbose=verbose-2,
		)		
		nparams_mhca = sum(length.(Flux.params(tfdb.mhca)))
		printlog(
			"$(size(x_ln1)) -> $(size(x_mhca))[$(size(cattn))]; $(nparams_mhca) params\n";
			context="MultiHeadCrossAttention()",
			type=:INFO,
			level=3,
			verbose=verbose-2,
		)
		nparams_ln1 = sum(length.(Flux.params(tfdb.ln1)))
		printlog(
			"$(size(x_mha)) -> $(size(x_ln1)); $(nparams_ln1) params\n";
			context="LayerNorm()",
			type=:INFO,
			level=3,
			verbose=verbose-2,
		)
		nparams_mha = sum(length.(Flux.params(tfdb.mha)))
		printlog(
			"$(size(x)) -> $(size(x_mha))[$(size(attn))]; $(nparams_mha) params\n";
			context="MultiHeadSelfAttention()",
			type=:INFO,
			level=3,
			verbose=verbose-2,
		)

		#superstructure
		printlog(
			"$(size(x)) -> $(size(x_ln3))[$(size(attn))$(size(cattn))]; $(tfdb.nparams) params\n";
			context="TransformerDecoderBlock()",
			type=:INFO,
			level=2,
			verbose=verbose-1,
		)
		
	end

	return x_ln3, attn, cattn
end
Flux.@functor TransformerDecoderBlock

#Masked Autoencoder Decoder
#--------------------------
"""
	- custom layer implementing the decoder part of a Masked Autoencoder

	Fields
	------
		- `pp` (trainable)
			- `PatchProjection`
			- layer to project patches onto vectors
		- `lpe` (trainable)
			- `LearnablePositionEncoding`
			- layer to add learnable position encodings
		- `tfdb` (trainable)
			- `AbstractVector{TransformerDecoderBlock}`
			- `n_blocks` instances of `TransformerDecoderBlock`
			- the number of `TransformerDecoderBlock` the model shall have
		- `n_heads` (non-trainable)
			- `Int`
			- number of heads in all of the `TransformerDecoderBlock`
			- only needed for size-casting
		- `seq_len` (non-trainable)
			- `Int`
			- sequence length of the input to the `MAEDecoder`
			- only for convenence (later reference)
		- `nparams`
			- `Int`
			- number of learnable parameters

	Methods
	-------
		- `MAEDecoder()`

	Dependencies
	------------
		- `Flux`

	Comments
	--------
"""
struct MAEDecoder
	pp::PatchProjection
	lpe::LearnablePositionEncoding	
	tfdb::AbstractVector{TransformerDecoderBlock}
	n_heads::Int	#only for formatting in forward pass
	seq_len::Int	#only for convenience (later reference)
	nparams::Int
end
"""
	- function to initialize a `MAEDecoder` layer

	Parameters
	----------
		- `size_patch`
			- `Tuple{Int,Int}`
			- dimensions of the patches in the training set
			- has to have form `(X,Y)`
		- `seq_len`
			- `Int`
			- length of the individual sequences		
		- `d_model`
			- `Int`
			- model dimension
			- i.e. hidden dimension of the model
			- applied to each of `n_blocks` `TransformerDecoderBlock`
		- `n_blocks`
			- `Int`
			- number of `TransformerDecoderBlock` the `MAEDecoder` shall have
		- `d_head`
			- `Int`
			- head dimension
			- total number of neurons distributed across attention heads in each `TransformerDecoderBlock`
			- has to be divisible by `n_heads`
			- applied to each of the`n_blocks` `TransformerDecoderBlock`
		- `n_heads`
			- `Int`
			- number of attention heads to use in each `TransformerDecoderBlock`
			- applied to each of `n_blocks` `TransformerDecoderBlock`
		- `d_ff`
			- `Int`
			- dimension of the feedforward block following the multi head attention in each `TransformerDecoderBlock`
			- applied to each of `n_blocks` `TransformerDecoderBlock`
		- `act_ff`
			- `Function`, optional
			- activation function to use in the feedforward block in each `TransformerDecoderBlock`
			- applied to each of `n_blocks` `TransformerDecoderBlock`
			- the default is `Flux.relu`
		- `mha_dropout`
			- `Float32`, optional
			- dropout in the multi-head attention in each `TransformerDecoderBlock`
			- applied to each of `n_blocks` `TransformerDecoderBlock`
			- the default is `0.`
		- `mhca_dropout`
			- `Float32`, optional
			- dropout in the multi-head cross attention in each `TransformerDecoderBlock`
			- applied to each of `n_blocks` `TransformerDecoderBlock`
			- the default is `0.`

	Raises
	------

	Returns
	-------
		- `MAEDecoder`
			- initialized instance of `MAEDecoder`

	Comments
	--------	
"""
function MAEDecoder(
	size_patch::Tuple{Int,Int}, seq_len::Int,
	d_model::Int,
	n_blocks::Int,
	d_head::Int, n_heads::Int,
	d_ff::Int;
	act_ff::Function=Flux.relu,
	mha_dropout::AbstractFloat=0.,
	mhca_dropout::AbstractFloat=0.,
	)
	pp = PatchProjection(size_patch, d_model)
	lpe = LearnablePositionEncoding(d_model, seq_len)
	tfdb = [
		TransformerDecoderBlock(
			d_model,
			d_head, n_heads,
			d_ff,
			act_ff=act_ff,
			mha_dropout=mha_dropout,
			mhca_dropout=mhca_dropout
		)
	for _ in 1:n_blocks]
	nparams = sum(length.(Flux.params(pp, lpe, tfdb...)))


	return MAEDecoder(pp, lpe, tfdb, n_heads, seq_len, nparams)
end
"""
	- forward pass of `MAEDecoder`
	
	Parameters
	----------
		- `x`
			- `AbstractArray`
			- input data to be propagated through the layer
		- `x_enc`
			- `AbstractArray{Float32}`
			- output of a transformer encoder block (`TransformerEncoderBlock`)
		- `mha_mask`
			- `BitMatrix`, `Nothing`, optional
			- multi-head attention mask in `Flux.MultiHeadAttention()` of each `TransformerDecoderBlock`
			- the default is `nothing`
				- no mask applied
		- `mhca_mask_dec`
			- `BitMatrix`, `Nothing`, optional
			- multi-head cross-attention mask in `Flux.MultiHeadAttention()` of each `TransformerDecoderBlock`
			- the default is `nothing`
				- no mask applied		
		- `train`
			- `Bool`, optional
			- whether the model is in training mode or not
			- important for gradient computation, as `Zygote` cannot deal with user defined functions in `Flux.gradient()`
			- setting to `true` will remove all custom function calls
			- the default is `false`
		- `verbose`
			- `Int`, optional
			- verbosity level
			- the default is `0`
	
	Raises
	------

	Returns
	-------
		- `x_maed`
			- `AbstractArray{Float32,3}`
			- `x` after propagating through the `MAEDecoder`
		- `attns`
			- `AbstractArray{Float32,5}`
			- self-attention maps of each `TransformerEncoderBlock`
				- each element contains the self-attention maps for each sample and each head
			- has size `(seq_len,seq_len,n_heads,n_samples,n_blocks)`
		- `cattns`
			- `AbstractArray{Float32,5}`
			- cross-attention maps of each `TransformerEncoderBlock`
				- each element contains the cross-attention maps for each sample and each head
			- has size `(seq_len_enc,seq_len_dec,n_heads,n_samples,n_blocks)`

	Comments
	--------
"""
function (maed::MAEDecoder)(
	x::AbstractArray{Float32}, x_enc::AbstractArray{Float32},
	mha_mask::Union{BitMatrix,Nothing}=nothing, mhca_mask::Union{BitMatrix,Nothing}=nothing;
	train::Bool=false,
	verbose::Int=0,
	)::Tuple{AbstractArray{Float32,3},AbstractArray{Float32,5},AbstractArray{Float32,5}}

	X, Y, seq_len_dec, batch_size = size(x)
	_, seq_len_enc, batch_size = size(x_enc)
	n_blocks = size(maed.tfdb,1)
	n_heads = maed.n_heads

	#projection and embeddin
	x_maed = maed.pp(x;			train=train, verbose=verbose)
	x_maed = maed.lpe(x_maed, 	train=train, verbose=verbose)	
	
	#propagation through all `TransformerDecoderBlock`
	attns	= Flux.Zygote.Buffer(Array{Float32,5}(undef, seq_len_dec, seq_len_dec, n_heads, batch_size, n_blocks))		#write to buffer to allow mutable operation
	cattns 	= Flux.Zygote.Buffer(Array{Float32,5}(undef, seq_len_enc, seq_len_dec,	n_heads, batch_size, n_blocks))		#write to buffer to allow mutable operation
	for i in eachindex(maed.tfdb)
		#pass through `TransformerDecoderBlock` `i`
		x_maed, attn, cattn = maed.tfdb[i](x_maed, x_enc, mha_mask, mhca_mask; train=train, verbose=verbose)
		
		#store attention maps
		attns[:,:,:,:,i] = attn
		cattns[:,:,:,:,i] = cattn
	end

	#summary
	if ~train
		printlog(
			"$(size(x)) -> $(size(x_maed)); $(maed.nparams) params\n";
			context="MAEDecoder()",
			type=:INFO,
			level=1,
			verbose=verbose,
		)			
	end

	return x_maed, copy(attns), copy(cattns)	#copy makes object immutable again
end
Flux.@functor MAEDecoder (pp, lpe, tfdb,)

#Masked Autoencoder Output
#-------------------------
"""
	- custom Masked Autoencoder output layer
	- interprets `MAEDecoderBlock` output

	Fields
	------
		- `ff` (trainable)
			- `Flux.Chain`
			- feedforward block receiving the output and projecting it onto probabilities
		- `size_patch` (non-trainable)
			- `Tuple{Int,Int}`
			- dimensions of the patches in the training set
			- has to have form `(X,Y)`
			- used to reshape the output of `ff`
		- `nparams`
			- `Int`
			- number of learnable parameters

	Methods
	-------
		- `MAEOut()`

	Dependencies
	------------
		- `Flux`

	Comments
	--------
	
"""
struct MAEOut
	ff::Flux.Chain
	size_patch::Tuple{Int,Int}
	nparams::Int
end
"""
	- function to initialize a `MAEOut` layer

	Parameters
	----------
		- `size_patch`
			- `Tuple{Int,Int}`
			- dimensions of the patches in the training set
			- has to have form `(X,Y)`
		- `d_model`
			- `Int`
			- model dimension
			- i.e. hidden dimension of the model
			
	Raises
	------

	Returns
	-------
		- `MAEOut`
			- initialization instance of `MAEOut`

	Comments
	--------
"""
function MAEOut(
	size_patch::Tuple{Int,Int}, d_model::Int;
	)::MAEOut

	X,Y = size_patch
	ff = Flux.Chain(
		Flux.Dense(d_model => X*Y),
		# Flux.sigmoid,	#for binary pixel values
		# Flux.softmax,
	)
	nparams = sum(length.(Flux.params(ff)))
	
	return MAEOut(ff, size_patch, nparams)
end
"""
	- forward pass of `MAEOut`
	
	Parameters
	----------
		- `x`
			- `AbstractArray{Float32}`
			- input data to be propagated through the layer
			- output of a MAE decoder (`MAEDecoder`)
		- `train`
			- `Bool`, optional
			- whether the model is in training mode or not
			- important for gradient computation, as `Zygote` cannot deal with user defined functions in `Flux.gradient()`
			- setting to `true` will remove all custom function calls
			- the default is `false`		
		- `verbose`
			- `Int`, optional
			- verbosity level
			- the default is `0`
	
	Raises
	------

	Returns
	-------
		- `x_out`
			- `AbstractArray`
			- `x` after propagating through the `MAEOut` block

	Comments
	--------
"""
function (maeo::MAEOut)(
	x::AbstractArray{Float32};
	train::Bool=false,
	verbose::Int=0,
	)::AbstractArray{Float32}
	
	#get sizes for reshape
	d_model, seq_len, batch_size = size(x)

	#propagate
	x_ff = maeo.ff(x)
	
	#reshape to be comparable with input images
	x_out = reshape(x_ff, maeo.size_patch..., seq_len, batch_size)

	#summary
	if ~train && verbose > 0
		nparams_ff = sum(length.(Flux.params(maeo.ff)))
		printlog(
			"$(size(x)) -> $(size(x_ff)); $(nparams_ff) params\n";
			context="Dense()",
			type=:INFO,
			level=2,
			verbose=verbose-2,
		)
		nparams_rs = 0
		printlog(
			"$(size(x_ff)) -> $(size(x_out)); $(nparams_rs) params\n";
			context="reshape()",
			type=:INFO,
			level=2,
			verbose=verbose-2,
		)
		printlog(
			"$(size(x)) -> $(size(x_out)); $(maeo.nparams) params\n";
			context="MAEOut()",
			type=:INFO,
			level=1,
			verbose=verbose,
		)
	end

	return x_out
end
Flux.@functor MAEOut (ff,)


#utilities
#---------
"""
    - function to save a MAE so it can be loaded with `mae_load()`
    - saves the complete `Structs` of `enc` and `dec` in a jld2-file with 3 keys
		- `"maee"`
		- `"maed"`
		- `"maeo"`

    Parameters
    ----------
		- `maee`
			- `MAEEncoder`
			- encoder part of the model
		- `maed`
			- `MAEDecoder`
			- decoder part of the model
		- `maeo`
			- `MAEOut`
			- output block of the model
    
    Raises
    ------

    Returns
    -------

    Dependencies
    ------------
        - `Flux`
        - `JLD2`

    Comments
    --------
        - requires `MAEEncoder`, `MAEDecoder`, and `MAEOut` to be defined!
"""
function mae_save(
    maee::MAEEncoder, maed::MAEDecoder, maeo::MAEOut,
    filename::String;
    )
    JLD2.jldsave(filename; maee=maee, maed=maed, maeo=maeo)

end

"""
    - function to load a MAE saved with `mae_save()`

    Parameters
    ----------
        - `filename`
            - `String`
            - file to load the data from
            - has to be a jld2-file with 3 keys
                - `"maee"`
				- `"maed"`
				- `"maeo"`
    
    Raises
    ------

    Returns
    -------
		- `maee`
			- `MAEEncoder`
			- encoder part of the model
		- `maed`
			- `MAEDecoder`
			- decoder part of the model
		- `maeo`
			- `MAEOut`
			- output block of the model

    Dependencies
    ------------
        - `JLD2`

    Comments
    --------
        - requires `MAEEncoder`, `MAEDecoder`, and `MAEOut` to be defined!
"""
function mae_load(
    filename::String,
    )::Tuple{MAEEncoder, MAEDecoder, MAEOut}

    maee = JLD2.load(filename, "maee")
    maed = JLD2.load(filename, "maed")
    maeo = JLD2.load(filename, "maeo")
    
    return maee, maed, maeo
end

"""
	- function computing the loss of the model given some targets

    Parameters
    ----------
        - `src`
            - input to the `MAEEncoder`
        - `trg`
            - input to the `MAEDecoder`
        - `y_true`
            - ground truth the `MAE` output shall be compared to
		- `mha_mask_enc`
			- `BitMatrix`, `Nothing`, optional
			- multi-head attention mask in the `MAEDecoder`
			- the default is `nothing`
				- no mask applied
		- `mha_mask_dec`
			- `BitMatrix`, `Nothing`, optional
			- multi-head attention mask in the `MAEDecoder`
			- the default is `nothing`
				- no mask applied
		- `mhca_mask_dec`
			- `BitMatrix`, `Nothing`, optional
			- multi-head cross-attention mask in the `MAEDecoder`
			- the default is `nothing`
				- no mask applied
		- `train`
			- `Bool`, optional
			- whether the model is in training mode or not
			- important for gradient computation, as `Zygote` cannot deal with user defined functions in `Flux.gradient()`
			- setting to `true` will remove all custom function calls
			- the default is `false`

	Raises
	------

	Returns
	-------
		- `err`
			- `Float32`
			- computed loss

	Dependencies
	------------

	Comments
	--------
"""
function loss(
    src::AbstractArray{Float32,4}, trg::AbstractArray{Float32,4}, y_true::AbstractArray{Float32,4},
    maee::MAEEncoder, maed::MAEDecoder, maeo::MAEOut;
	mha_mask_enc::Union{BitMatrix,Nothing}=nothing,
	mha_mask_dec::Union{BitMatrix,Nothing}=nothing, mhca_mask_dec::Union{BitMatrix,Nothing}=nothing,
	train::Bool=false,
    )::Float32
    enc_out, enc_attn 				= maee(src, 		 mha_mask_enc; 					train=train, verbose=0)
    dec_out, dec_attn, dec_cattn 	= maed(trg, enc_out, mha_mask_dec, mhca_mask_dec; 	train=train, verbose=0)
	y_pred 							= maeo(dec_out; 									train=train, verbose=0)

	seq_len_enc = maee.seq_len
	seq_len_dec = maed.seq_len

    # err = Flux.Losses.mse(y_pred, y_true)
	err = mean((y_pred[:,:,:,seq_len_enc:end] .- y_true[:,:,:,seq_len_enc:end]) .^ 2)	#mse (only consider masked tokens)
	# err = mean((y_pred .- y_true) .^ 2)	#mse
	# err = Flux.Losses.binarycrossentropy(y_pred, y_true)
	# err = Flux.Losses.binarycrossentropy(y_pred[:,:,:,seq_len_enc:end], y_true[:,:,:,seq_len_enc:end])	#(only consider masked tokens)
    return err
end

"""
	- function to display a model summary
	- will store the summary in a `DataFrame` for convenience

	Parameters
	----------
		- `maee`
			- `MAEEncoder`
			- Encoder part of the complete Vision Transformer
		- `maed`
			- `MAEDecoder`
			- Decoder part of the complete Vision Transformer
		- `maeo`
			- `MAEOut`
			- Output part of the complete Vision Transformer
		- `x_enc`
			- `AbstractArray{Float32,4}`
			- pseudo-array to be used for propagating through the model
			- serves as input for the encoder
			- it is recommended to pass something like `zeros(Float32, X, Y, seq_len_enc, 1)` for performance sake
		- `x_dec`
			- `AbstractArray{Float32,4}`
			- pseudo-array to be used for propagating through the model
			- serves as input for the decoder
			- it is recommended to pass something like `zeros(Float32, X, Y, seq_len_dec, 1)` for performance sake
		- `mha_mask_enc`
			- `BitMatrix`, `Nothing`, optional
			- multi-head attention mask to be used in `MAEEncoder`
			- the default is `nothing`
				- no mask applied
		- `mha_mask_dec`
			- `BitMatrix`, `Nothing`, optional
			- multi-head attention mask to be used in `MAEDecoder`
			- the default is `nothing`
				- no mask applied
		- `mhca_mask_dec`
			- `BitMatrix`, `Nothing`, optional
			- multi-head cross-attention mask to be used in `MAEDecode`
			- the default is `nothing`
				- no mask applied	
		- `verbose`
			- `Int`, optional
			- verbosity level
			- controls how many sublayers to show
			- the default is `2`
				- only top layers shown

	Raises
	------

	Returns
	-------
		- `df`
			- `DataFrames.DataFrame`
			- contains model-summary in the form of a table

	Dependencies
	------------
		- `DataFrames`
		- `DataFramesMeta`

	Comments
	--------
"""
function model_summary(
	maee::MAEEncoder, maed::MAEDecoder, maeo::MAEOut,
	x_enc::AbstractArray{Float32,4}, x_dec::AbstractArray{Float32,4};
	mha_mask_enc::Union{BitMatrix,Nothing}=nothing,
	mha_mask_dec::Union{BitMatrix,Nothing}=nothing, mhca_mask_dec::Union{BitMatrix,Nothing}=nothing,
	show::Bool=true, verbose::Int=2,
	)::DataFrames.DataFrame

    #run model with exemplary data (store output in file)
    open("temp.txt", "w") do f
        redirect_stdout(f) do
            #propagate with verbosity
            x_maee, _		= maee(x_enc, 			mha_mask_enc; 					train=false, verbose=verbose)
            x_maed, _, _ 	= maed(x_dec, x_maee, 	mha_mask_dec, mhca_mask_dec;	train=false, verbose=verbose)
            _ 				= maeo(x_maed;											train=false, verbose=verbose)
        end
    end

	#read output from file
	f = open("temp.txt", "r")
	lines = readlines(f)
	close(f)
	if show println(join(lines, "\n")) end	#print if requested
	rm("temp.txt")
	
	#extract information for table
    layer_parents = getfield.(match.(r"^\s+(?!=Info)", lines), :match)
    layer_names = getfield.(match.(r"(?<=INFO\()\w+", lines), :match)
    layer_inputs = getfield.(match.(r"(?<=: )\([\d\, ]+\)", lines), :match)
    layer_outputs = getfield.(match.(r"(?<=-> )[\d\, \(\)\[\]]+", lines), :match)
	layer_params = getfield.(match.(r"\d+(?=\ params)", lines), :match)
    
	#modify names to show relation to parents
	layer_names = layer_parents .* layer_names
    layer_names = replace.(layer_names, r"\ {4}(?=\w)"=>"|-")
    layer_names = replace.(layer_names, r"\ "=>"~")
    
	#store layers in table
	df = DataFrames.DataFrame(
        "Name"          => layer_names,
        "Input Shape"   => layer_inputs,
        "Output Shape"  => layer_outputs,
        "Parameters"    => layer_params,
    )

    # nparams_enc = sum(length.(Flux.params(maee)))
    # nparams_dec = sum(length.(Flux.params(maed)))
    # nparams_out = sum(length.(Flux.params(maeo)))
    # @transform!(df, @byrow :Parameters = :Name == "|-MAEEncoder" ? nparams_enc : :Parameters)
    # @transform!(df, @byrow :Parameters = :Name == "|-MAEDecoder" ? nparams_dec : :Parameters)
    # @transform!(df, @byrow :Parameters = :Name == "|-MAEOut" ? nparams_out : :Parameters)

    df = df[end:-1:1,:]	#reverse such that order makes more sense

	return df
end

"""
	- function to predict future frames using some MAE

	Parameters
	----------
		- `x`
			- `AbstractArray{Float32,4}`
			- input series to be used as a basis for the prediction
			- has to have size `(X,Y,seq_len_in,n_samples)`
		- `seq_len`
			- `Int`
			- number of new tokens to predict
		- `maee`
			- `MAEEncoder`
			- Encoder part of the complete Vision Transformer
		- `maed`
			- `MAEDecoder`
			- Decoder part of the complete Vision Transformer
		- `maeo`
			- `MAEOut`
			- Output part of the complete Vision Transformer
		- `verbose`
			- `Int`, optional
			- verbosity level
			- the default is `0`

	Raises
	------
		- `AssertionError`
			- in case the input `x` has too short sequences

	Returns
	-------
		- `x_pred`
			- `AbstractArray{Float32,4}`
			- predicted tokens
			- has size `(X,Y,seq_len,n_samples)`

	Dependencies
	------------

	Comments
	--------
"""
function mae_predict(
	x::AbstractArray{Float32,4},
	seq_len::Int,
	maee::MAEEncoder, maed::MAEDecoder, maeo::MAEOut;
	mha_mask_enc::Union{BitMatrix,Nothing}=nothing,
	mha_mask_dec::Union{BitMatrix,Nothing}=nothing, mhca_mask_dec::Union{BitMatrix,Nothing}=nothing,	
	verbose::Int=0,
	)::AbstractArray{Float32,4}

	seq_len_enc = maee.seq_len
	seq_len_dec = maed.seq_len

	#get sizes
	X, Y, seq_len_in, n_samples = size(x)
	
	#init output
	x_pred = Array{Float32,4}(undef, X, Y, seq_len, n_samples)

	
	begin #get enc and dec inputs
		enc_in = x[:,:,1:seq_len_enc,:]
		dec_in = x[:,:,:,:]
	end
	for token in 1:seq_len
		
		#propagate through model
		enc_out, _ 		= maee(enc_in, 			mha_mask_enc; 				train=false, verbose=verbose)
		dec_out, _, _ 	= maed(dec_in, enc_out, mha_mask_dec, mhca_mask_dec;train=false, verbose=verbose)
		out_out			= maeo(dec_out; 									train=false, verbose=verbose)
	
		#get current prediction
		x_pred_i = out_out[:,:,seq_len_enc+1:seq_len_enc+1,:]

		#update model inputs for next token to predict
		enc_in = cat(enc_in, dec_in[:,:,1:1,:]; dims=3)[:,:,2:end,:]		#shift encoder input to next token
		dec_in = cat(dec_in, x_pred_i; dims=3)[:,:,2:end,:]					#shift decoder input to next token #add new output to the end
		
		#add to prediction
		x_pred[:,:,token,:] = x_pred_i[:,:,1,:]
	end
	
	return x_pred
end


#%%examples
begin
	# using LinearAlgebra

	# x_enc = rand(Float32,16,16,8,5)	#(X,Y,S,B)
	# x_dec = rand(Float32,16,16,32,5)	#(X,Y,S,B)
	# X, Y, seq_len_enc, batch_size = size(x_enc)
	# X, Y, seq_len_dec, batch_size = size(x_dec)
	# seq_len_out = 16
	# d_model = 64
	# n_blocks = 2
	# d_head = 64
	# n_heads = 2
	# d_ff = 256
	# act_ff = Flux.relu
	# mha_dropout = 0.
	# mhca_dropout = 0.
	# verbose = 2

	# # # mha = Flux.MultiHeadAttention(d_model => d_head => d_model)
	# # # x_mha = ones(Float32, d_model, seq_len_dec, 5)
	# # # mask = LinearAlgebra.triu!(trues(seq_len_dec, seq_len_dec))
	# # # mask[end-seq_len_out:end,:] .= false	#mask last steps
	# # # # mask = Flux.make_causal_mask(x_mha; dims=2)
	# # # x_mha, attn = mha(x_mha; mask=mask)
	# # # println(size(x_mha), size(attn), size(mask))


	# # # h1 = heatmap(attn[:,:,1,1], colorbar=false)
	# # # h2 = heatmap(mask)
	# # # p = plot(h1, h2, layout=(1,2), yflip=false, xflip=true)
	# # # display(p)

	# maee = MAEEncoder(
	# 	(X,Y), seq_len_enc,
	# 	d_model,
	# 	n_blocks,
	# 	d_head, n_heads,
	# 	d_ff;
	# 	act_ff=act_ff,
	# 	mha_dropout=mha_dropout,
	# )
	# maed = MAEDecoder(
	# 	(X,Y), seq_len_dec,
	# 	d_model,
	# 	n_blocks,
	# 	d_head, n_heads,
	# 	d_ff;
	# 	act_ff=act_ff,
	# 	mha_dropout=mha_dropout,
	# 	mhca_dropout=mhca_dropout,
	# )
	# maeo = MAEOut(
	# 	(X,Y), d_model,
	# )

	# # ps = Flux.params(maee, maed, maeo)
	# # println(length(size.(ps)))

	# mha_mask_enc 	= LinearAlgebra.triu!(trues(seq_len_enc, seq_len_enc))
	# mha_mask_dec 	= LinearAlgebra.triu!(trues(seq_len_dec, seq_len_dec))
	# mhca_mask_dec 	= LinearAlgebra.triu!(trues(seq_len_enc, seq_len_dec))
	# # eout, eattn 		= maee(zeros(Float32, X, Y, seq_len_enc, 1), mha_mask_enc; verbose=2)
	# # dout, dattn, dcattn = maed(zeros(Float32, X, Y, seq_len_dec, 1), mha_mask_enc, mha_mask_dec, mhca_mask_dec; verbose=2)

	# # # # p = plot(layout=@layout[[a b] c])
	# # # h1 = heatmap(eattn[ :,:,2,1,2])
	# # # h2 = heatmap(dattn[	:,:,2,1,2])
	# # # h3 = heatmap(dcattn[:,:,2,1,2])
	# # # p_ = plot()
	# # # plot(h1, h2, h3, layout=@layout[a b;_ c{0.5w} _])

	# df_model = model_summary(
	# 	maee, maed, maeo,
	# 	zeros(Float32, X, Y, seq_len_enc, 2),
	# 	zeros(Float32, X, Y, seq_len_dec, 2);
	# 	mha_mask_enc=mha_mask_enc,
	# 	mha_mask_dec=mha_mask_dec,
	# 	mhca_mask_dec=mhca_mask_dec,
	# 	verbose=4,
	# )

	# println(df_model)
end