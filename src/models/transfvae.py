# from src.models.vae import Sampling,VAE

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
from src.utils.metrics import masked_mse
def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    # def __init__(self, vocab_size, d_model):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
        
        # Giacomo: this adds a sequential embedding layer to learn a representation of the tokens.
        # Commented because of two problems:
        #   1) There is no masking for the padding as in Embedding (where mask_zero can be True)
        #   2) Incorrect representation, since each "group" has different filters (it's not a single embedding)
        # self.embedding = Sequential([])
        # self.embedding.add(Conv1D(filters=16,kernel_size=50,padding="same",groups=16,data_format="channels_first"))
        # self.embedding.add(Conv1D(filters=16,kernel_size=50,padding="same",groups=16,data_format="channels_first"))

        self.embedding = Masking()
        self.pos_encoding = positional_encoding(length=16, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)
    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    # self.pos_embedding = PositionalEmbedding(
    #     vocab_size=vocab_size, d_model=d_model)
    self.pos_embedding = PositionalEmbedding(d_model=d_model)
    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # TODO: add a new type of embedding suitable for time series
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
    # Add dropout.
    # x = self.dropout(x)
    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()
    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)
    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)
    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)
    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores
    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x

class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.pos_embedding = PositionalEmbedding(d_model=d_model) # I deleted vocab size
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
    x = self.dropout(x)
    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)
    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x

class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            vocab_size=input_vocab_size,
                            dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            vocab_size=target_vocab_size,
                            dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
        # Metrics:
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        
        # Test metrics
        self.test_total_loss_tracker = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            # self.reconstruction_loss_tracker,
            # self.kl_loss_tracker,
            self.test_total_loss_tracker,
            # self.test_reconstruction_loss_tracker,
            # self.test_kl_loss_tracker
        ]
        
    def train_step(self, data):
        with tf.GradientTape() as tape:
            inputs,logits,_,_ = self(data)
            total_loss,reconstruction_loss,kl_loss = self.compute_loss(inputs, logits)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        # self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)
        # self.total_loss_tracker.merge_state([self.reconstruction_loss_tracker,self.kl_loss_tracker]) #update_state(total_loss)
        # self.total_loss_tracker.update_state([self.reconstruction_loss_tracker.result(),self.kl_loss_tracker.result()],[self.alpha,self.beta])
        return {
            "loss": self.total_loss_tracker.result()}
        
    def test_step(self, data):
        inputs,logits,_,_ = self(data)
        total_loss,reconstruction_loss,kl_loss = self.compute_loss(inputs, logits)
         # TODO: check if test trackers are correct here
        # self.test_reconstruction_loss_tracker.update_state(reconstruction_loss)
        # self.test_kl_loss_tracker.update_state(kl_loss)
        self.test_total_loss_tracker.update_state(total_loss)
        # self.test_total_loss_tracker.update_state([self.test_reconstruction_loss_tracker.result(),self.test_kl_loss_tracker.result()],[self.alpha,self.beta])
        # self.test_total_loss_tracker.merge_state([self.test_reconstruction_loss_tracker,self.test_kl_loss_tracker])#.update_state(total_loss)
        return {
            "loss": self.test_total_loss_tracker.result(),
            # "reconstruction_loss": self.test_reconstruction_loss_tracker.result(),
            # "kl_loss": self.test_kl_loss_tracker.result(),
        }
    def compute_loss(self, inputs, logits):
        data = inputs #["data_morpho"]
        reconstruction = logits
        reconstruction_loss = masked_mse(data,reconstruction)
        # mask = data!=0
        # mask = tf.cast(mask, dtype=data.dtype)
        # squared_difference = tf.math.squared_difference(reconstruction, data)
        # # reconstruction_loss = tf.reduce_mean(tf.reduce_sum(squared_difference,axis=[1,2]),axis=0)
        # squared_difference *= mask
        # # Do the division by sum of mask values to make the mean over only the non-padding values 
        # reconstruction_loss = tf.reduce_sum(tf.reduce_sum(squared_difference,axis=[1,2]),axis=0)/tf.reduce_sum(mask)
        return reconstruction_loss, tf.zeros((1)), tf.zeros((1)) # TODO: Replace reconstruction_loss with total loss for VAE
    
    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        # context, x  = inputs
        context, x = inputs["data_morpho"], inputs["data_morpho"]
        context = self.encoder(context)  # (batch_size, context_len, d_model)
        x = self.decoder(x, context)  # (batch_size, target_len, d_model)
        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
        try:
        # Drop the keras mask, so it doesn't scale the losses/metrics.
        # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass
        # Return the final output and the attention weights.
        z_mean = tf.zeros(shape=(10))
        z_log_var = tf.zeros(shape=(10))
        return inputs["data_morpho"], logits, z_mean, z_log_var

# class TransfVAE(VAE):
#     # TODO: overritw methods of VAE! The structure is too different,
#     # It is not possible to define only encoder and decoder without modifying the vae structure
#     def __init__(self, encoder, decoder, alpha=1, beta=1, **kwargs):
#        super().__init__(encoder, decoder, alpha, beta, **kwargs)

if __name__ == '__main__':
    sample_encoder = Encoder(num_layers=4,
                         d_model=512,
                         num_heads=8,
                         dff=2048,
                         vocab_size=8500)
    sample_encoder.summary()
    