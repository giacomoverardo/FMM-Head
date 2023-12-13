# from src.models.vae import Sampling,VAE

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
from keras.models import Model, Sequential
from keras.layers import BatchNormalization, Conv1D, Dense, Flatten, Conv1DTranspose, \
                        Dropout, InputLayer, AveragePooling1D, Reshape, \
                        UpSampling1D, Masking, Add
from keras.activations import relu,sigmoid,tanh
from typing import List,Tuple
from src.models.vae import VAE, Sampling
from src.models.baseFMMAE import Base_FMM_AE
# from vae import VAE, Sampling

# Code implemented from description of CVAE (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0260612)
class Cvae_block(Model):
    def __init__(self, kernel_size: int, num_filters:int=1, dropout_rate:float=0.1, add_avg_pool:bool=True, add_skip_connection:bool=True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.add_avg_pool = add_avg_pool
        self.batch_norm1 = BatchNormalization()
        self.add_skip_connection = add_skip_connection
        if(add_skip_connection):
            self.convinput = Conv1D(filters=self.num_filters,kernel_size=1,padding="same")
        self.conv1 = Conv1D(filters=self.num_filters,kernel_size=self.kernel_size,padding="same")
        self.conv2 = Conv1D(filters=self.num_filters,kernel_size=self.kernel_size,padding="same")
        self.avg_pool  = AveragePooling1D(padding="same")
        self.batch_norm2 = BatchNormalization()
        self.batch_norm3 = BatchNormalization()
        self.dropout = Dropout(rate=self.dropout_rate)
        self.add = Add()
        self.activation = tanh

    def call(self, inputs, training):
        return self.__call__(inputs=inputs,training=training)

    def __call__(self, inputs,training=None, mask=None) -> tf.Tensor:
        x = inputs
        x = self.batch_norm1(inputs=x,training=training)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.batch_norm2(inputs=x,training=training)
        x = self.activation(x)
        x = self.conv2(x)
        if(self.add_skip_connection):
            convinputs = self.convinput(inputs)
            convinputs = self.batch_norm3(inputs=convinputs,training=training)
            x = self.add([x,convinputs])
        if(self.add_avg_pool):
            x = self.avg_pool(x)
        x = self.activation(x)
        if(training):
            x = self.dropout(x)
        return x #{"output":x}

class Cvae_block_decoder(Cvae_block):
    def __init__(self, kernel_size: int, num_filters: int = 1, dropout_rate: float = 0.1, add_avg_pool: bool = True) -> None:
        super().__init__(kernel_size, num_filters, dropout_rate, add_avg_pool,add_skip_connection=True)
        self.conv1 = Conv1DTranspose(self.num_filters,self.kernel_size,padding="same")
        self.conv2 = Conv1DTranspose(self.num_filters,self.kernel_size,padding="same")
        self.avg_pool = UpSampling1D()

class Cvae_block_encoder(Cvae_block):
    def __init__(self, kernel_size: int, num_filters: int = 1, dropout_rate: float = 0.1, add_avg_pool: bool = True) -> None:
        super().__init__(kernel_size, num_filters, dropout_rate, add_avg_pool)

class Cvae_coder(Model):
    def __init__(self,input_shape:Tuple, sequential_part = None, filter_size_list:List =[19,19,19,19,19,19,9,9,9], num_filters_list = [16,16,16,32,48,64,64,80,80,80],
            add_avg_pool_list =  [1,1,1,1,1,0,0,0,0]) -> None: 
        super().__init__()
        self.sequential_part = Sequential() if sequential_part is None else sequential_part
        if(input_shape):
            self.sequential_part.add(InputLayer(input_shape=input_shape))
        for kernel_size,num_filters,add_pool in zip(filter_size_list,num_filters_list,add_avg_pool_list):
            self.sequential_part.add(self.block_type(kernel_size = kernel_size,num_filters=num_filters,add_avg_pool=add_pool))

    def __call__(self, inputs, training=None, mask=None):
        return self.sequential_part(inputs=inputs,training=training)

    def get_sequential_output_shape(self, sample):
        out = self.sequential_part(sample)
        return out.shape
    
    def call(self, inputs, training):
        return self.__call__(inputs=inputs,training=training)

class Cvae_encoder(Cvae_coder):

    def __init__(self, input_shape: Tuple, 
                 latent_dim:int,
                 sequential_part:Sequential=None, 
                 filter_size_list: List = [19, 19, 19, 19, 19, 19, 9, 9, 9], 
                 num_filters_list=[16, 16, 16, 32, 48, 64, 64, 80, 80], 
                 add_avg_pool_list=[1, 1, 1, 1, 1, 0, 0, 0, 0]) -> None:
        self.block_type = Cvae_block_encoder
        if(sequential_part is None):
            sequential_part = Sequential()
            sequential_part.add(InputLayer(input_shape=input_shape))
            sequential_part.add(Masking(mask_value=0.)) #Mask zero values
        super().__init__(input_shape, sequential_part, filter_size_list, num_filters_list, add_avg_pool_list)
        self.latent_dim = latent_dim
        if(latent_dim is not None):
            self.sequential_part.add(Flatten())
            self.z_mean_layer = layers.Dense(latent_dim, name="z_mean")
            self.z_log_var_layer = layers.Dense(latent_dim, name="z_log_var")
            self.sampling = Sampling()

    def get_sequential_output_shape(self, sample):
        out_shape_after_flatten =  super().get_sequential_output_shape(sample)
        out_shape_before_flatten =  self.sequential_part.layers[-2].output_shape
        return out_shape_before_flatten
    
    def latent_step(self,x):
        if(self.latent_dim is not None):
            z_mean = self.z_mean_layer(x)
            z_log_var = self.z_log_var_layer(x)
            z = self.sampling([z_mean,z_log_var])
            return {"z_mean":z_mean, "z_log_var": z_log_var, "z":z}
        else:
            return x
    
    def __call__(self, inputs, training=None, mask=None):
        encoded_vec = super().__call__(inputs, training, mask)
        return self.latent_step(encoded_vec)
    
class Cvae_decoder(Cvae_coder):
    def __init__(self, input_shape: Tuple, latent_dim:int, 
                 filter_size_list: List = [9, 9, 9, 19, 19, 19, 19, 19, 19], 
                 num_filters_list=[80, 80, 80, 64, 64, 48, 32, 16, 16], 
                 add_avg_pool_list=[0, 0, 0, 0, 1, 1, 1, 1, 1], 
                 out_shape=None) -> None:
        self.block_type = Cvae_block_decoder
        sequential_part = Sequential()
        sequential_part.add(InputLayer(input_shape=[latent_dim]))
        sequential_part.add(layers.Dense(np.prod(input_shape[1:])))
        sequential_part.add(layers.Reshape(target_shape=input_shape[1:]))
        super().__init__(input_shape=None, sequential_part = sequential_part, filter_size_list=filter_size_list, num_filters_list=num_filters_list, add_avg_pool_list=add_avg_pool_list)
        self.sequential_part.add(AveragePooling1D(pool_size = 16, data_format="channels_first"))
        
        # self.input_layer = InputLayer(input_shape=[latent_dim])
        # self.dense_tail = layers.Dense(np.prod(input_shape[1:]))
        # self.tail_reshape = layers.Reshape(target_shape=input_shape[1:])
        if(out_shape):
            # self.sequential_part.add(Reshape(target_shape=(-1,)))
            self.sequential_part.add(Flatten())
            num_dense_nodes = np.prod(out_shape)
            self.sequential_part.add(Dense(num_dense_nodes)) 
            self.sequential_part.add(Reshape(target_shape=out_shape))
            # self.dense_head = Dense(num_dense_nodes)
            # self.flatten = Flatten()
            # self.reshape = Reshape(target_shape=out_shape)
            
    def call(self, inputs, training):
        return self.__call__(inputs=inputs,training=training)
    
    def __call__(self, inputs, training=None, mask=None):
        x = self.sequential_part(inputs=inputs,training=training)
        return x

class CVAE(VAE):
    def __init__(self, latent_dim, input_seq_len, num_features, batch_size,alpha=1, beta=1, f1=19, f2=9, num_warmup_epochs=500, **kwargs):
        self.f1 = f1
        self.f2 = f2
        input_shape = [input_seq_len,num_features]
        encoder = Cvae_encoder( input_shape=input_shape,
                                latent_dim=latent_dim, 
                                filter_size_list = [f1, f1, f1, f1, f1, f1, f2, f2, f2], 
                                num_filters_list=  [16, 16, 16, 32, 48, 64, 64, 80, 80], 
                                add_avg_pool_list= [ 1,  1,  1,  1,  1,  0,  0,  0,  0])
        decoder_input_shape = encoder.get_sequential_output_shape(np.zeros([batch_size]+input_shape)) 
        decoder = Cvae_decoder( input_shape=decoder_input_shape, 
                                latent_dim=latent_dim,
                                filter_size_list = [f2, f2, f2, f1, f1, f1, f1, f1, f1], 
                                num_filters_list=  [80, 80, 64, 64, 48, 32, 16, 16, 16], #[80, 80, 80, 64, 64, 48, 32, 16, 16], 
                                add_avg_pool_list= [0,   0,  0,  0,  1,  1,  1,  1,  1],
                                out_shape=input_shape)
        self.num_warmup_epochs = num_warmup_epochs
        self.need_warmup = True if num_warmup_epochs>0 else False
        super().__init__(encoder, decoder, alpha, beta, **kwargs)

class FMM_CAE(Base_FMM_AE):
    def __init__(self, f1, f2, 
                 num_leads, seq_len, max_omega, batch_size, split_ecg, 
                 reconstruction_loss_weight, coefficient_loss_weight, 
                 num_warmup_epochs, coeffs_properties_dict, *args, **kwargs):
        super().__init__(num_leads, seq_len, max_omega, batch_size, 
                         split_ecg, reconstruction_loss_weight, coefficient_loss_weight,
                         num_warmup_epochs, coeffs_properties_dict, *args, **kwargs)
        self.global_avg_pooling = tf.keras.layers.Flatten()#tf.keras.layers.GlobalAveragePooling1D()
        self.f1 = f1
        self.f2 = f2
        input_shape = [seq_len,num_leads]
        self.encoder = Cvae_encoder( input_shape=input_shape,
                                latent_dim=None, 
                                filter_size_list = [f1, f1, f1, f1, f1, f1, f2, f2, f2], 
                                num_filters_list=  [16, 16, 16, 32, 48, 64, 64, 80, 80], 
                                add_avg_pool_list= [ 1,  1,  1,  1,  1,  0,  0,  0,  0])
        
if __name__ == '__main__':
    pass