from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import GlobalAveragePooling1D, ReLU, Conv1D, Dense, Conv2D, Flatten, Input, \
									BatchNormalization, MaxPooling2D, \
									Dropout, InputLayer, Embedding, LSTM, GlobalAveragePooling2D, Conv1DTranspose, \
									MultiHeadAttention, LayerNormalization,Reshape, LeakyReLU, Masking
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.activations import softmax, sigmoid, relu
from tensorflow.keras.losses import categorical_crossentropy, mse
from src.models.vae import *
from typing import List, Tuple, Dict
import src.utils.general_functions as gf 

def get_transformer_model_1d_series_classification(input_shape, n_classes=None):
	""" https://huggingface.co/keras-io/timeseries_transformer_classification"""
	# model = TFAutoModelForSequenceClassification.from_pretrained("keras-io/timeseries_transformer_classification")
	def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
		# Attention and Normalization
		x = MultiHeadAttention(
			key_dim=head_size, num_heads=num_heads, dropout=dropout
		)(inputs, inputs)
		x = Dropout(dropout)(x)
		x = LayerNormalization(epsilon=1e-6)(x)
		res = x + inputs

		# Feed Forward Part
		x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
		x = Dropout(dropout)(x)
		x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
		x = LayerNormalization(epsilon=1e-6)(x)
		return x + res
	def build_model(
		input_shape,
		head_size,
		num_heads,
		ff_dim,
		num_transformer_blocks,
		mlp_units,
		dropout=0,
		mlp_dropout=0,
	):
		inputs = Input(shape=input_shape)
		x = inputs
		for _ in range(num_transformer_blocks):
			x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

		x = GlobalAveragePooling1D(data_format="channels_first")(x)
		for dim in mlp_units:
			x = Dense(dim, activation="relu")(x)
			x = Dropout(mlp_dropout)(x)
		if(n_classes):
			outputs = Dense(n_classes, activation="softmax")(x)
		else:
			outputs = x #My change to let it work for anomaly detection
		return Model(inputs, outputs)
	model = build_model(
		input_shape,
		head_size=256,
		num_heads=4,
		ff_dim=4,
		num_transformer_blocks=4,
		mlp_units=[128],
		mlp_dropout=0.4,
		dropout=0.25,
	)
	return model

def create_LSTM_model_adaptive_FD(input_shape, max_in_value, input_length, lstm_units=256):
	"""For Shakespeare dataset, we consider a two-layer LSTM
	classifier containing 256 hidden units preceded by an 8-
	dimensional embedding layer. The embedding layer takes a
	sequence of 80 characters as input, and the output is a class
	label between 0 and 52"""
	model = tf.keras.Sequential([
		InputLayer(input_shape=input_shape),
		Embedding(input_dim=max_in_value, output_dim=8, input_length=input_length),
		LSTM(units=lstm_units, return_sequences=True), #input_shape=(160,1)
		LSTM(units=lstm_units, return_sequences=True),
		Dense(max_in_value)
	])
	return model

def get_autoencoder_model_1D(input_shape,kernel_size:int=15, strides:int=1):
	"""https://keras.io/examples/timeseries/timeseries_anomaly_detection/
	"""
	model = Sequential(
		[
			Input(shape=input_shape),
			Conv1D(
				filters=32, kernel_size=kernel_size, padding="same", strides=strides, activation="relu"
			),
			Dropout(rate=0.2),
			Conv1D(
				filters=16, kernel_size=kernel_size, padding="same", strides=strides, activation="relu"
			),
			Dropout(rate=0.2),
			Conv1D(
				filters=8, kernel_size=kernel_size, padding="same", strides=strides, activation="relu"
			),
			Dropout(rate=0.2),
			Flatten(),
			Dense(1024),
			Dropout(rate=0.2),
			Dense(128),
			Dropout(rate=0.2),
			Dense(1250*8),
			Dropout(rate=0.2),
			Reshape((1250, 8)),
			Conv1DTranspose(
				filters=8, kernel_size=kernel_size, padding="same", strides=strides, activation="relu"
			),		
			Dropout(rate=0.2),
			Conv1DTranspose(
				filters=16, kernel_size=kernel_size, padding="same", strides=strides, activation="relu"
			),
			Dropout(rate=0.2),
			Conv1DTranspose(
				filters=32, kernel_size=kernel_size, padding="same", strides=strides, activation="relu"
			),
			Conv1DTranspose(filters=1, kernel_size=kernel_size, padding="same"),
		]
	)
	# print(model.summary())
	return model

def get_conv_model1D(input_shape, num_classes)->Model:
    input_layer = Input(input_shape)

    conv1 = Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)

    conv2 = Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)

    conv3 = Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)

    gap = GlobalAveragePooling1D()(conv3)

    output_layer = Dense(num_classes, activation="softmax")(gap)

    return Model(inputs=input_layer, outputs=output_layer)

def get_dense_network(numNodesPerLayer:List[int], 
		      			inputShape:Tuple[int,...]=None, 
						dropoutRate:List[float]=0.0, 
						activation:List[str]="relu",
						initializer: List[str]="glorot_normal",
						addInFlatten:bool=False,
						maskValue:bool=None)->Sequential:
	numLayers = len(numNodesPerLayer)
	dropoutRateList = gf.scalar_to_list(dropoutRate,numLayers)
	activationList = gf.scalar_to_list(activation,numLayers)
	initializerList = gf.scalar_to_list(initializer,numLayers)
	model = tf.keras.models.Sequential()
	if(inputShape is not None):
		model.add(Input(shape=inputShape))
	if(addInFlatten):
		model.add(Flatten())
	if(maskValue is not None):
		model.add(Masking(maskValue))
	for i,(numNodes, droprate, act, init) in enumerate(zip(numNodesPerLayer,dropoutRateList,activationList,initializerList)):
		if(i==(numLayers-1)):
			pass
		else:
			model.add(Dense(units=numNodes, activation=act, kernel_initializer=init))
			if(droprate>0.0):
				model.add(Dropout(rate=droprate))
	model.add(Dense(units=numNodesPerLayer[-1],kernel_initializer=initializerList[-1])) #Linear activation for last layer
	return model

def get_classification_convolutional_model_1d(input_shape:List[int], num_filters_per_layer:List[int], strides_list:List[int], 
                                              kernel_size_list:List[int],
                                              padding:str, output_size:int, transpose:bool=False,
											latent_dim:int=None,model_type:str=None,add_flatten:bool=True)->Sequential:
	layer_type = Conv1DTranspose if transpose else Conv1D
	model = Sequential()
	if(model_type=="decoder"):
		model.add(InputLayer(input_shape=[latent_dim]))
		model.add(Dense(np.prod(input_shape[1:])))
		model.add(Reshape(target_shape=input_shape[1:]))
	else:
		model.add(InputLayer(input_shape=input_shape))
	for num_filters,kernel_size,strides in zip(num_filters_per_layer,kernel_size_list,strides_list):
		model.add(layer_type(filters=num_filters,kernel_size=kernel_size,strides=strides, padding=padding))
		model.add(LeakyReLU(alpha=0.01))
	if(add_flatten):
		model.add(Flatten())
	if(output_size):
		model.add(Dense(units=output_size))
	if(model_type=="encoder"):
		inputs = tf.keras.Input(shape=input_shape)
		x = model(inputs)
		z_mean = Dense(latent_dim, name="z_mean")(x)
		z_log_var = Dense(latent_dim, name="z_log_var")(x)
		z = Sampling()([z_mean,z_log_var])
		encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
		return encoder
	return model

if __name__ == '__main__':
	# model = get_transformer_model_1d_series_classification()
	# model.summary()
	numClasses=5
	seqLen=1250
	numSeq = 1000
	batchSize = 32
	epochs=1
	x = np.random.random(size=(numSeq,seqLen))
	y = np.random.randint(low=0,high=numClasses,size=(numSeq,))
	# x.reshape((x.shape[0], x.shape[1], 1))
	x = tf.expand_dims(x, axis=-1)
	model = get_conv_model1D(input_shape=x.shape[1:],num_classes=numClasses)
	model.compile(
		optimizer="adam",
		loss="sparse_categorical_crossentropy",
		metrics=["sparse_categorical_accuracy"],
	)
	history = model.fit(
		x,
		y,
		batch_size=batchSize,
		epochs=epochs,
		# callbacks=callbacks,
		validation_split=0.2,
		verbose=1,
	)
	plt.figure()
	plt.plot(history.history["sparse_categorical_accuracy"])
