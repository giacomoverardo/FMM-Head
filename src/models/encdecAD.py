from src.models.baseAE import BaseAE
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Masking, RepeatVector, \
                                        Flatten, TimeDistributed, LSTMCell, Identity
from tensorflow.keras.models import Sequential
import numpy as np
# from src.models.fmm_ae import FMM_AE_regression_circular_ang, FMM_head_ang
import tensorflow as tf
from src.models.baseFMMAE import Base_FMM_AE
from src.utils.metrics import masked_mse,mse_timeseries, weighted_mean_squared_error, \
                            circular_weighted_mean_square_error

class CustomLSTMCell(tf.keras.layers.LSTMCell):
    def __init__(self, num_units, activation=tf.tanh, **kwargs):
        super(CustomLSTMCell, self).__init__(num_units, activation=activation, **kwargs)
        # Define trainable variables for multiplicative and bias factors
        self.multiplicative_factor = self.add_weight(shape=(self.units,1), initializer=tf.keras.initializers.GlorotUniform(), name='multiplicative_factor',trainable=True)
        self.bias_factor = self.add_weight(shape=(1, 1), initializer='zeros', name='bias_factor',trainable=True)

    def call(self, inputs, states, training=None):
        # Unpack the states into cell state and output state
        state_h, state_c = states
        # Modify the output with the linear transformation
        predicted_output = tf.matmul(state_h,self.multiplicative_factor) + self.bias_factor
        # Choose the input based on the "training" flag
        if(training):
            # Use the external input (real sequence) during training
            input_to_super_call = inputs
        else:
            # Use the output from the previous timestep during inference
            input_to_super_call = predicted_output
        # Call the parent class (LSTMCell) call method to perform standard LSTM computations
        output, new_states = super(CustomLSTMCell, self).call(input_to_super_call, [state_h, state_c], training=training)
        return predicted_output, new_states


class EncDecAD(BaseAE):
    def __init__(self, enc_units, dec_units, sequence_length, **kwargs):
        super().__init__(**kwargs)
        self.num_enc_layers = len(enc_units)
        self.num_dec_layers = len(dec_units)
        # In default encdecad, there is only one layer per enc/dec with same size
        assert self.num_dec_layers==1, self.num_enc_layers==1
        assert enc_units[0]==dec_units[0]
        self.num_units = enc_units[0]
        self.sequence_length = sequence_length
        self.encoder = LSTM(self.num_units, return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.RNN(CustomLSTMCell(self.num_units), 
                                           return_sequences=True, 
                                           return_state=True, 
                                           zero_output_for_mask=True, 
                                           go_backwards=True)
        self.output_dense = Dense(units=1,activation="linear")                
        self.masking = Masking(0.0)

    def call_github(self,inputs,training):
        x = inputs["inputs"]
        mask = self.masking.compute_mask(x)
        x = self.masking(x)
        x , state_h, state_c = self.encoder(x,training=training,mask=mask)
        enc_state = [state_h, state_c]
        # Initialize the outputs.
        y = tf.TensorArray(
            element_shape=(x.shape[0], x.shape[2]),
            size=x.shape[1],
            dynamic_size=False,
            dtype=tf.float32,
            clear_after_read=False
        )
        # Update the encoder states.
        he = self.encoder(x)
        # Initialize the decoder states.
        hd = tf.identity(he)
        cd = tf.zeros(shape=(tf.shape(inputs)[0], self.units))
        # Update the decoder states.
        for t in tf.range(start=inputs.shape[1] - 1, limit=-1, delta=-1):
            y = y.write(index=t, value=self.outputs(hd))
            hd, [hd, cd] = self.decoder(states=[hd, cd], inputs=x[:, t, :] if training else y.read(index=t))
        # Return the outputs.
        return tf.transpose(y.stack(), (1, 0, 2))
        # del output_sequence._keras_mask
        return {"data":inputs["inputs"], "predicted_data":output_sequence}
    
    def call(self, inputs, training):
        x = inputs["inputs"]
        mask = self.masking.compute_mask(x)
        x = self.masking(x)
        x , state_h, state_c = self.encoder(x,training=training,mask=mask)
        enc_state = [state_h, state_c]
        output_sequence_reversed, final_output, final_state = self.decoder(inputs=inputs["inputs"], initial_state=enc_state,training=training,)# mask=mask)
        output_sequence = tf.reverse(output_sequence_reversed,axis=[1])
        return {"data":inputs["inputs"], "predicted_data":output_sequence}

class FMM_EncDecAD(Base_FMM_AE):
    def __init__(self, enc_units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_enc_layers = len(enc_units)
        assert self.num_enc_layers==1
        self.num_units = enc_units[0]
        self.encoder = Sequential([Masking(mask_value=0.0),LSTM(self.num_units, return_sequences=False, return_state=False)])
        self.global_avg_pooling = Identity()
        
if __name__ == '__main__':
    pass