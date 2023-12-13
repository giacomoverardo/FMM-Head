import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
from keras.models import Model, Sequential
from keras.layers import BatchNormalization, Conv1D, Dense, Flatten, Conv1DTranspose, \
                        Dropout, InputLayer, AveragePooling1D, Reshape, \
                        UpSampling1D, Masking, Add, SeparableConv1D
from keras.activations import relu
from typing import List,Tuple
from src.models.vae import VAE, Sampling
from src.utils.metrics import masked_mse,mse_timeseries
from src.models.baseAE import BaseAE
from src.utils.general_functions import *

class Conv_AE(BaseAE):
    def __init__(self, num_filters_list, kernel_sizes_list, strides_list, padding="same",
                activation="sigmoid",dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = len(num_filters_list)
        assert self.num_layers%2==0 #Half of filters goes to encoder, half to decoder
        self.num_filters_list=num_filters_list
        self.kernel_sizes_list = scalar_to_list(kernel_sizes_list,self.num_layers)
        self.strides_list = scalar_to_list(strides_list,self.num_layers)
        self.padding=padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        #Layers
        self.masking = Masking()
        self.dropout = [Dropout(dropout_rate) for _ in range(self.num_layers)]
        self.enc_layers = [
            Conv1D(filters=self.num_filters_list[i],kernel_size=self.kernel_sizes_list[i],
                   strides=self.strides_list[i],padding=padding,activation=activation)
            for i in range(0,int(self.num_layers/2))]
        self.num_enc_layers = len(self.enc_layers)
        assert (self.num_enc_layers*2 )==self.num_layers
        self.dec_layers = [
            Conv1DTranspose(filters=self.num_filters_list[i],kernel_size=self.kernel_sizes_list[i],
                   strides=self.strides_list[i],padding=padding,activation=activation)
            for i in range(int(self.num_layers/2),self.num_layers)]
        # i = self.num_layers-1
        # self.dec_layers.append(Conv1DTranspose(filters=self.num_filters_list[i],kernel_size=self.kernel_sizes_list[i],
        #            strides=self.strides_list[i],padding=padding)) 
        self.num_dec_layers = len(self.dec_layers)
        assert (self.num_dec_layers*2 )==self.num_layers
        self.final_layer = Conv1D(filters=kwargs["num_leads"],kernel_size=self.kernel_sizes_list[-1],
                   strides=1,padding=padding) #No activation in last layer
        

    def call(self, inputs):
        x = inputs["inputs"]
        x = self.masking(x)
        for i in range(self.num_enc_layers):
            x = self.enc_layers[i](x)
            x = self.dropout[i](x)
        for i in range(self.num_dec_layers-1):
            x = self.dec_layers[i](x)
            x = self.dropout[i+self.num_enc_layers](x)
        x = self.dec_layers[-1](x)
        x = self.final_layer(x)
        try:
        # Drop the keras mask, so it doesn't scale the losses/metrics.
            del x._keras_mask
        except AttributeError:
            pass
        return inputs["inputs"], x, tf.zeros(shape=(2)), tf.zeros(shape=(2))

    
if __name__=='__main__':
    pass

