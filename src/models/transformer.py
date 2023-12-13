# from src.models.vae import Sampling,VAE

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
from keras.models import Model, Sequential
from keras.layers import BatchNormalization, Conv1D, Dense, Flatten, Conv1DTranspose, \
                        Dropout, InputLayer, AveragePooling1D, Reshape, \
                        UpSampling1D, Masking, Add, SeparableConv1D, TimeDistributed
from keras.activations import relu
from typing import List,Tuple
from src.utils.metrics import masked_mse,mse_timeseries
from src.models.baseFMMAE import Base_FMM_AE
from src.utils.nn import Squeeze

# def positional_encoding(length, depth):
#     depth = depth/2

#     positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
#     depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

#     angle_rates = 1 / (10000**depths)         # (1, depth)
#     angle_rads = positions * angle_rates      # (pos, depth)

#     pos_encoding = np.concatenate(
#         [np.sin(angle_rads), np.cos(angle_rads)],
#         axis=-1) 

#     return tf.cast(pos_encoding, dtype=tf.float32)

# def positional_encoding(length, depth, n=10000):
#     P = np.zeros((length, depth))
#     for k in range(length):
#         for i in np.arange(int(depth/2)):
#             denominator = np.power(n, 2*i/depth)
#             P[k, 2*i] = np.sin(k/denominator)
#             P[k, 2*i+1] = np.cos(k/denominator)
#     return tf.cast(P, dtype=tf.float32)

def positional_encoding_tf(length, depth, max_len, n=10000):
    half_depth = depth/2
    positions = tf.range(length,dtype=tf.float32)[:, tf.newaxis]     # (seq, 1)
    depths = tf.range(half_depth,dtype=tf.float32)[tf.newaxis, :]/half_depth   # (1, depth)
    angle_rates = 1 / (n**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)
    pad_len = max_len - tf.cast(length,tf.int32)
    padding = tf.zeros((pad_len,depth),dtype=tf.float32)
    pos_encoding = tf.concat([tf.math.sin(angle_rads), tf.math.cos(angle_rads)], axis=-1) 
    pos_encoding = tf.concat([pos_encoding, padding], axis=0)  
    return pos_encoding 

def positional_encoding_ecg(lengths, max_length, depth, n=10000):
    pos_encoding_list = []
    for len in lengths:
        pos_encoding = positional_encoding_tf(length=len, depth=depth, max_len=max_length, n=n)
        pos_encoding_list.append(pos_encoding)
    pos_encoding_stack = tf.stack(pos_encoding_list)
    return pos_encoding_stack

# def positional_encoding_ecg(batch_size, lengths, max_length, depth, n=10000):
#     P = tf.zeros((batch_size, max_length, depth))
#     P = []
#     for len in lengths:     
#         for k in range(len):
#             for i in tf.range(int(depth/2)):
#                 denominator = tf.math.pow(n,2*i/depth)
#                 psin = tf.math.sin(k/denominator)
#                 pcos = tf.math.cos(k/denominator)
#                 vec_encoding = tf.concat([psin,pcos],1)
#                 P.append(vec_encoding)
#     P_stack = tf.stack(P)
#     return P_stack

class PositionalEmbedding(tf.keras.layers.Layer):
    # def __init__(self, vocab_size, d_model):
    def __init__(self, d_model, max_len):
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

        self.masking = Masking()
        self.embedding = Sequential([])
        # self.embedding.add(Masking())
        self.embedding.add(Dense(self.d_model,activation="linear"))
        # self.embedding.add(Dense(self.d_model,activation="sigmoid"))
        self.embedding.add(Dropout(0.1))
        # self.pos_encoding = positional_encoding(length=1000, depth=d_model)
        self.pos_encoding_ecg = positional_encoding_ecg(np.arange(0,max_len,1,int), max_len, d_model, n=10000)
    def compute_mask(self, *args, **kwargs):
        return self.masking.compute_mask(*args, **kwargs)

    def call(self, x, sizes, training):
        length = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]
        # [np.min(np.where(r==0)[0]) for r in x._keras_mask.numpy()]
        x = self.masking(inputs=x, training=training)
        x = self.embedding(inputs=x, training=training)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x = x + self.pos_encoding[tf.newaxis, :length, :]
        selected_rows = tf.gather(self.pos_encoding_ecg, tf.cast(sizes,tf.int32), axis=0)
        expanded_row_indices = tf.expand_dims(sizes, axis=-1)
        col_indices_tensor = tf.range(self.d_model)
        selected_rows_and_columns = tf.gather(selected_rows, col_indices_tensor, axis=-1)
        x = x + selected_rows_and_columns
        # x = x + self.pos_encoding_ecg[tf.cast(sizes,tf.int32)]
        # x = x + positional_encoding_ecg(lengths=sizes,max_length=length,depth=self.d_model)
        return x

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = layers.LayerNormalization() #layers.BatchNormalization(axis=-1) #tf.keras.layers.BatchNormalization() #tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

# class CrossAttention(BaseAttention):
#   def call(self, x, context, training):
#     attn_output, attn_scores = self.mha(
#         query=x,
#         key=context,
#         value=context,
#         return_attention_scores=True)
#     # Cache the attention scores for plotting later.
#     self.last_attn_scores = attn_scores
#     x = self.add([x, attn_output])
#     x = self.layernorm(inputs=x,training=training)
#     return x

class GlobalSelfAttention(BaseAttention):
  def call(self, x, training, mask):
    attention_mask = mask[:, tf.newaxis, :]
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        attention_mask=attention_mask)
    # mask = self.mha.compute_mask(,mask)
    x = self.add([x, attn_output])
    x = self.layernorm(inputs=x,training=training,)
    return x

# class CausalSelfAttention(BaseAttention):
#   def call(self, x, training):
#     attn_output = self.mha(
#         query=x,
#         value=x,
#         key=x,
#         use_causal_mask = True)
#     x = self.add([x, attn_output])
#     x = self.layernorm(inputs=x,training=training)
#     return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = layers.LayerNormalization() #layers.BatchNormalization(axis=-1) #tf.keras.layers.BatchNormalization() #tf.keras.layers.LayerNormalization()

  def call(self, x, training):
    x = self.add([x, self.seq(inputs=x,training=training)])
    x = self.layer_norm(inputs=x,training=training)
    return x

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def compute_mask(self, inputs, mask=None):
     return mask
 
  def call(self, x, training, mask):
    x = self.self_attention(x=x,training=training, mask=mask)
    x = self.ffn(x=x,training=training)
    return x

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads,
               dff, vocab_size, max_len, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    # self.pos_embedding = PositionalEmbedding(
    #     vocab_size=vocab_size, d_model=d_model)
    self.pos_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    # self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs, training):
    # TODO: add a new type of embedding suitable for time series
    # `x` is token-IDs shape: (batch, seq_len)
    sizes = inputs["sizes"]
    x = inputs["inputs"]
    mask = self.pos_embedding.compute_mask(x)
    x = self.pos_embedding(x=x, sizes=sizes, training=training)  # Shape `(batch_size, seq_len, d_model)`.
    # Add dropout.
    # x = self.dropout(x)
    for i in range(self.num_layers):
      x = self.enc_layers[i](x=x,training=training, mask=mask)

    return x  # Shape `(batch_size, seq_len, d_model)`.

class Multi_size_encoder(Encoder):
   def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super(Encoder,self).__init__()
        #Compared to encoder, d_model, num_heads and dff are now iterables
        len_d_model = len(d_model)-1
        len_num_heads = len(num_heads)
        len_dff = len(dff)
        assert len_d_model==len_num_heads
        assert len_num_heads==len_dff

        self.d_model_list = d_model
        self.dff_list=dff 
        self.num_heads_list = num_heads
        self.num_layers = len_dff

        self.pos_embedding = PositionalEmbedding(d_model=d_model[0])
        self.enc_layers = [
            EncoderLayer(d_model=d_model[i+1],
                        num_heads=num_heads[i],
                        dff=dff[i],
                        dropout_rate=dropout_rate)
            for i in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
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
  def __init__(self, num_layers, d_model, num_heads, dff, vocab_size,
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

class Bert_ecg(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                input_vocab_size, target_vocab_size, max_len, dropout_rate=0.1, **kwargs):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            vocab_size=input_vocab_size,
                            dropout_rate=dropout_rate,
                            max_len=max_len)
        self.kwargs = kwargs
        # self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
        #                     num_heads=num_heads, dff=dff,
        #                     vocab_size=target_vocab_size,
        #                     dropout_rate=dropout_rate)

        self.final_layer = Sequential() 
        final_layer_nodes = self.kwargs.get("final_layer_nodes",None)
        if(final_layer_nodes):
            for num_nodes in final_layer_nodes:
                self.final_layer.add(tf.keras.layers.Dense(num_nodes,activation="tanh"))
        # self.final_layer.add(tf.keras.layers.Dense(512,activation="tanh"))
        # self.final_layer.add(tf.keras.layers.Dense(256,activation="tanh")) #Modify to dff to 256 to make it good
        self.final_layer.add(tf.keras.layers.Dense(target_vocab_size))
        self.global_avg_pooling = Flatten() #tf.keras.layers.GlobalAveragePooling1D() 
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
            loss_dict = self.compute_loss(self(inputs=data,training=True))
            total_loss_mean = tf.reduce_mean(loss_dict["total_loss"])
        grads = tape.gradient(total_loss_mean, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss_dict["total_loss"])
        return {
            "loss": self.total_loss_tracker.result()}
        
    def test_step(self, data):
        loss_dict = self.compute_loss(self(inputs=data,training=False))
        self.test_total_loss_tracker.update_state(loss_dict["total_loss"])
        return {
            "loss": self.test_total_loss_tracker.result(),
        }
        
    def compute_loss(self, inputs):
        data = inputs["data"]
        predicted_data = inputs["predicted_data"]
        loss = masked_mse(data,predicted_data)
        return {"total_loss":loss}
    
    def call(self, inputs, training):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        x = inputs["inputs"]
        sizes = inputs["sizes"]
        x = self.encoder(inputs={"inputs":x,"sizes":sizes},training=training)  # (batch_size, context_len, d_model)
        # x = self.decoder(x, context)  # (batch_size, target_len, d_model)
        # Final linear layer output.
        # x = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
        # x = self.global_avg_pooling(x)
        # x = tf.expand_dims(x,-1)
        x = self.final_layer(inputs=x,training=training)
        try:
        # Drop the keras mask, so it doesn't scale the losses/metrics.
        # b/250038731
            del x._keras_mask
        except AttributeError:
            pass
        return {"data":inputs["inputs"], "predicted_data":x}

class Attention_AE(Bert_ecg):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1, **kwargs):
       super(Bert_ecg,self).__init__()
       self.kwargs = kwargs
       
       
class Attention_AE(Bert_ecg):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1, **kwargs):
        super().__init__(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate, **kwargs)
        self.encoder = Multi_size_encoder(num_layers=None, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            vocab_size=input_vocab_size,
                            dropout_rate=dropout_rate)
               
    # def __init__2(self, d_model, num_heads, dff,
    #             input_vocab_size, target_vocab_size, dropout_rate=0.1, **kwargs):
    #     super().__init__()
        # self.encoder = Multi_size_encoder(num_layers=None, d_model=d_model,
        #                     num_heads=num_heads, dff=dff,
        #                     vocab_size=input_vocab_size,
        #                     dropout_rate=dropout_rate)

        # self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
        #                     num_heads=num_heads, dff=dff,
        #                     vocab_size=target_vocab_size,
        #                     dropout_rate=dropout_rate)

        # self.final_layer = Sequential() 
        # # self.final_layer.add(tf.keras.layers.Dense(512,activation="tanh"))
        # self.final_layer.add(tf.keras.layers.Dense(64,activation="tanh"))
        # self.final_layer.add(tf.keras.layers.Dense(target_vocab_size))
        # self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling1D()
        # # Metrics:
        # self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        
        # # Test metrics
        # self.test_total_loss_tracker = keras.metrics.Mean(name="total_loss")

    # @property
    # def metrics(self):
    #     return [
    #         self.total_loss_tracker,
    #         # self.reconstruction_loss_tracker,
    #         # self.kl_loss_tracker,
    #         self.test_total_loss_tracker,
    #         # self.test_reconstruction_loss_tracker,
    #         # self.test_kl_loss_tracker
    #     ]
        
    # def train_step(self, data):
    #     with tf.GradientTape() as tape:
    #         inputs,logits,_,_ = self(data)
    #         total_loss,reconstruction_loss,kl_loss = self.compute_loss(inputs, logits)
    #         total_loss_mean = tf.reduce_mean(total_loss)
    #     grads = tape.gradient(total_loss_mean, self.trainable_weights)
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    #     # self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    #     # self.kl_loss_tracker.update_state(kl_loss)
    #     self.total_loss_tracker.update_state(total_loss)
    #     # self.total_loss_tracker.merge_state([self.reconstruction_loss_tracker,self.kl_loss_tracker]) #update_state(total_loss)
    #     # self.total_loss_tracker.update_state([self.reconstruction_loss_tracker.result(),self.kl_loss_tracker.result()],[self.alpha,self.beta])
    #     return {
    #         "loss": self.total_loss_tracker.result()}
        
    # def test_step(self, data):
    #     inputs,logits,_,_ = self(data)
    #     total_loss,reconstruction_loss,kl_loss = self.compute_loss(inputs, logits)
    #      # TODO: check if test trackers are correct here
    #     # self.test_reconstruction_loss_tracker.update_state(reconstruction_loss)
    #     # self.test_kl_loss_tracker.update_state(kl_loss)
    #     self.test_total_loss_tracker.update_state(total_loss)
    #     # self.test_total_loss_tracker.update_state([self.test_reconstruction_loss_tracker.result(),self.test_kl_loss_tracker.result()],[self.alpha,self.beta])
    #     # self.test_total_loss_tracker.merge_state([self.test_reconstruction_loss_tracker,self.test_kl_loss_tracker])#.update_state(total_loss)
    #     return {
    #         "loss": self.test_total_loss_tracker.result(),
    #         # "reconstruction_loss": self.test_reconstruction_loss_tracker.result(),
    #         # "kl_loss": self.test_kl_loss_tracker.result(),
    #     }
    # def compute_loss(self, inputs, logits):
    #     data = inputs #["data_morpho"]
    #     reconstruction = logits
    #     reconstruction_loss = masked_mse(data,reconstruction)
    #     # reconstruction_loss = mse_timeseries(data,reconstruction)

    #     return reconstruction_loss, tf.zeros((1)), tf.zeros((1)) # TODO: Replace reconstruction_loss with total loss for VAE
    
    # def call(self, inputs):
    #     # To use a Keras model with `.fit` you must pass all your inputs in the
    #     # first argument.
    #     # context, x  = inputs
    #     x = inputs["inputs"]
    #     x = self.encoder(x)  # (batch_size, context_len, d_model)
    #     # x = self.decoder(x, context)  # (batch_size, target_len, d_model)
    #     # Final linear layer output.
    #     # x = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    #     # x = self.global_avg_pooling(x)
    #     # x = tf.expand_dims(x,-1)
    #     x = self.final_layer(x)
    #     try:
    #     # Drop the keras mask, so it doesn't scale the losses/metrics.
    #     # b/250038731
    #         del x._keras_mask
    #     except AttributeError:
    #         pass
    #     # Return the final output and the attention weights.
    #     z_mean = tf.zeros(shape=(10))
    #     z_log_var = tf.zeros(shape=(10))
    #     return inputs["inputs"], x, z_mean, z_log_var
# class TransfVAE(VAE):
#     # TODO: overritw methods of VAE! The structure is too different,
#     # It is not possible to define only encoder and decoder without modifying the vae structure
#     def __init__(self, encoder, decoder, alpha=1, beta=1, **kwargs):
#        super().__init__(encoder, decoder, alpha, beta, **kwargs)

class Bert_ecg2(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                input_vocab_size, target_vocab_size, dropout_rate=0.1, **kwargs):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            vocab_size=input_vocab_size,
                            dropout_rate=dropout_rate)

        # self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
        #                     num_heads=num_heads, dff=dff,
        #                     vocab_size=target_vocab_size,
        #                     dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling1D()
        
    def train_step(self, data):
        with tf.GradientTape() as tape:
            inputs,logits,_,_ = self(data)
            total_loss = self.compute_loss(inputs, logits)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss}
        
    def test_step(self, data):
        inputs,logits,_,_ = self(data)
        total_loss = self.compute_loss(inputs, logits)
        return {
            "loss": total_loss
        }
    
    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        # context, x  = inputs
        x = inputs["inputs"]
        x = self.encoder(x)  # (batch_size, context_len, d_model)
        # x = self.decoder(x, context)  # (batch_size, target_len, d_model)
        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
        try:
        # Drop the keras mask, so it doesn't scale the losses/metrics.
        # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass
        logits = self.global_avg_pooling(logits)
        logits = tf.expand_dims(logits,-1)
        return logits

class FMM_Bert_ECG(Base_FMM_AE):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate,
                    num_leads, seq_len, max_omega, batch_size, split_ecg, 
                    reconstruction_loss_weight, coefficient_loss_weight, num_warmup_epochs, 
                    coeffs_properties_dict, *args, **kwargs):
        super().__init__(num_leads, seq_len, max_omega, batch_size, split_ecg, reconstruction_loss_weight, coefficient_loss_weight, num_warmup_epochs, coeffs_properties_dict, *args, **kwargs)
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            vocab_size=input_vocab_size,
                            dropout_rate=dropout_rate,
                            max_len=seq_len)
        self.global_avg_pooling = Sequential([TimeDistributed(Dense(1)), Squeeze()])
        self.encoder_input_type = "dict"
if __name__ == '__main__':
    pass
    