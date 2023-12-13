from src.models.baseAE import BaseAE
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Masking, RepeatVector, \
                                    Flatten, TimeDistributed, LSTMCell, Identity, \
                                    Dropout, GlobalAveragePooling1D, ReLU
from tensorflow.keras.models import Sequential
import numpy as np
# from src.models.fmm_ae import FMM_AE_regression_circular_ang, FMM_head_ang
import tensorflow as tf
from src.models.baseFMMAE import Base_FMM_AE
from src.utils.nn import Squeeze
class EcgNet(BaseAE):
    def __init__(self, enc_units, dec_units, sequence_length, num_features, dropout_rate, add_relu, **kwargs):
        super().__init__(**kwargs)
        self.num_enc_layers = len(enc_units)
        self.num_dec_layers = len(dec_units)
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.masking = Masking(mask_value=0.0)
        self.encoder = []
        self.encoder_dropout = [Dropout(dropout_rate) for i in range(self.num_enc_layers-1)]
        for i,enc_unit in enumerate(enc_units):
            self.encoder.append(LSTM(enc_unit, return_sequences=True))
        self.decoder = [] 
        self.decoder_dropout = [Dropout(dropout_rate) for i in range(self.num_dec_layers-1)]
        for i,dec_unit in enumerate(dec_units):
            self.decoder.append(LSTM(dec_unit, return_sequences=True))
        self.add_relu = add_relu
        if(add_relu):
            self.relu = tf.keras.activations.relu
        self.last_layer = TimeDistributed(Dense(num_features))
        

    def call(self, inputs, training):
        x = inputs["inputs"]
        x = self.masking(x)
        # Encoder part
        for i in range(self.num_enc_layers):
            x = self.encoder[i](inputs=x,training=training)
            if (i<self.num_enc_layers-1): # Only do relu and dropout if it's before the last layer
                if(self.add_relu):
                    x = self.relu(x)
                if(training):   # Do not apply dropout in test/validation
                    x = self.encoder_dropout[i](x)
        # Decoder part
        for i in range(self.num_dec_layers):
            x = self.decoder[i](inputs=x,training=training)
            if (i<self.num_dec_layers-1): # Only do relu and dropout if it's before the last layer
                if(self.add_relu):
                    x = self.relu(x)
                if(training):   # Do not apply dropout in test/validation
                    x = self.decoder_dropout[i](x)
        # Reconstruction
        x = self.last_layer(x)
        return {"data":inputs["inputs"], "predicted_data":x}

class FMM_EcgNet(Base_FMM_AE):
    def __init__(self, enc_units, sequence_length, num_features, dropout_rate, add_relu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_enc_layers = len(enc_units)
        self.encoder_layers = []
        for i,enc_unit in enumerate(enc_units):
            self.encoder_layers.append(LSTM(enc_unit, return_sequences=True))
            if(i<self.num_enc_layers-1):
                if(add_relu):
                    self.encoder_layers.append(ReLU())
                self.encoder_layers.append(Dropout(dropout_rate))
        self.encoder = Sequential(self.encoder_layers)
        self.global_avg_pooling = Sequential([TimeDistributed(Dense(1)), Squeeze()])
        self.sequence_length = sequence_length
        self.num_features = num_features

if __name__ == '__main__':
    pass