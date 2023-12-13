import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
from src.utils.metrics import masked_mse


class Sampling(layers.Layer):
    """Source: https://keras.io/examples/generative/vae/ """
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def get_conv_vae_encoder(latent_dim,input_shape=(28, 28, 1)):
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder

def get_conv_vae_decoder(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder

def get_lstm_vae_encoder(latent_dim,input_shape,num_lstm_layers:int=1,lstm_units_per_layer:int=32,
                        add_pool = False):
    encoder_inputs = keras.Input(shape=input_shape)
    x = encoder_inputs
    x = tf.keras.layers.Masking()(x)
    for i in range(num_lstm_layers):
        x = layers.LSTM(units=lstm_units_per_layer,
                        activation="tanh",
                        return_sequences=True,
                        recurrent_activation="sigmoid",
                        recurrent_dropout=0,
                        unroll=False,
                        use_bias=True)(x)
    if(add_pool):
        x = layers.MaxPool1D(pool_size=5,strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=16)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

def get_lstm_vae_decoder(latent_dim, sequence_length,num_lstm_layers:int=1,lstm_units_per_layer:int=32):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(sequence_length)(latent_inputs)
    x = layers.Reshape((sequence_length,1))(x)
    # x = layers.RepeatVector(sequence_length)(latent_inputs)
    for i in range(num_lstm_layers):
        x = layers.LSTM(lstm_units_per_layer, return_sequences=True,
                        activation="tanh",
                        recurrent_activation="sigmoid",
                        recurrent_dropout=0,
                        unroll=False,
                        use_bias=True)(x)
    decoder_outputs = layers.TimeDistributed(layers.Dense(1))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder

class SimpleMetric(keras.metrics.Metric):
    def __init__(self,name,dtype=None) -> None:
        super().__init__(name=name, dtype=dtype)
        # self.total = 0.0
    def update_state(self, values,weights):
        tf.math.m
        self.total = tf.math.add()
    def result(self):
        return tf.identity(self.total)


class VAE(keras.Model):
    def __init__(self, encoder, decoder, alpha=1.0, beta = 1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        # Train trackers
        # self.total_loss_tracker = keras.metrics.Metric(name="total_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        # Test trackers
        # self.test_total_loss_tracker = keras.metrics.Metric(name="total_loss")
        self.test_total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.test_reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.test_kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.alpha = alpha
        self.beta = beta
        # self.latent_dim = latent_dim
        # if(self.latent_dim):
        #     self.z_mean_layer = layers.Dense(latent_dim, name="z_mean")
        #     self.z_log_var_layer = layers.Dense(latent_dim, name="z_log_var")
        #     self.sampling = Sampling()
        #     encoder_output_flattened = encoder.layers[-1].output_shape
        #     encoder_output_before_flatten = encoder.layers[-2].output_shape
        #     self.decoder_dense_tail = layers.Dense(np.prod(encoder_output_before_flatten[1:]))
        #     self.decoder_reshape = layers.Reshape(target_shape=encoder_output_before_flatten[1:]) #TODO: avoid hard coding by taking encoder output shape (before flatten)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.test_total_loss_tracker,
            self.test_reconstruction_loss_tracker,
            self.test_kl_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss_dict = self.compute_loss(self(inputs=data,training=True))
            total_avg_loss = tf.reduce_mean(loss_dict["total_loss"])
        grads = tape.gradient(total_avg_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(loss_dict["reconstruction_loss"])
        self.kl_loss_tracker.update_state(loss_dict["kl_loss"])
        self.total_loss_tracker.update_state(loss_dict["total_loss"])
        # self.total_loss_tracker.merge_state([self.reconstruction_loss_tracker,self.kl_loss_tracker]) #update_state(total_loss)
        # self.total_loss_tracker.update_state([self.reconstruction_loss_tracker.result(),self.kl_loss_tracker.result()],[self.alpha,self.beta])
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def compute_loss(self, inputs):
        data,reconstruction,z_mean,z_log_var = \
                inputs["data"], inputs["predicted_data"], inputs["z_mean"], inputs["z_log_var"]
        # reconstruction_loss = tf.reduce_mean(tf.square(reconstruction - data), axis=-1)
        # reconstruction_loss = tf.reduce_mean(tf.square(reconstruction - data), axis=-1)
        kl_loss_vec = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(kl_loss_vec, axis=1) #tf.reduce_mean(tf.reduce_sum(kl_loss_vec, axis=1))
        # kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
        # reconstruction_loss = tf.reduce_sum(tf.math.squared_difference(reconstruction, data), axis=[1:tf.rank(reconstruction)])
        
        # Mask the padded values of the loss (the one where data is zero)
        # mask = data!=0
        # mask = tf.cast(mask, dtype=data.dtype)
        # squared_difference = tf.math.squared_difference(reconstruction, data)
        # # reconstruction_loss = tf.reduce_mean(tf.reduce_sum(squared_difference,axis=[1,2]),axis=0)
        # squared_difference *= mask
        # # Do the division by sum of mask values to make the mean over only the non-padding values 
        # reconstruction_loss = tf.reduce_sum(tf.reduce_sum(squared_difference,axis=[1,2]),axis=0)/tf.reduce_sum(mask)
        # # Check again the loss function : No problem: the fact that it's not between 0 and 1 is because we sum (not mean) over the squared errors for each time instant 
        # # [1:(tf.rank(reconstruction))]
        # # sum the two and average over batches
        reconstruction_loss = masked_mse(data,reconstruction)
        total_loss = self.alpha*reconstruction_loss + self.beta*kl_loss
        return {"total_loss":total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss":kl_loss}

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self,value):
        if(value<0.0):
            raise ValueError("Alpha should be greater or equal than 0")
        self._alpha = value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self,value):
        if(value<0.0):
            raise ValueError("Beta should be greater or equal than 0")
        self._beta = value

    def test_step(self, data):
        loss_dict = self.compute_loss(self(data,training=False))
         # TODO: check if test trackers are correct here
        self.test_reconstruction_loss_tracker.update_state(loss_dict["reconstruction_loss"])
        self.test_kl_loss_tracker.update_state(loss_dict["kl_loss"])
        self.test_total_loss_tracker.update_state(loss_dict["total_loss"])
        # self.test_total_loss_tracker.update_state([self.test_reconstruction_loss_tracker.result(),self.test_kl_loss_tracker.result()],[self.alpha,self.beta])
        # self.test_total_loss_tracker.merge_state([self.test_reconstruction_loss_tracker,self.test_kl_loss_tracker])#.update_state(total_loss)
        return {
            "loss": self.test_total_loss_tracker.result(),
            "reconstruction_loss": self.test_reconstruction_loss_tracker.result(),
            "kl_loss": self.test_kl_loss_tracker.result(),
        }

    def __call__(self, inputs, training=None, mask=None):
        data = inputs["inputs"]
        encoder_out = self.encoder(data,training=training)
        z, z_mean, z_log_var = encoder_out["z"],encoder_out["z_mean"],encoder_out["z_log_var"]
        decoder_out = self.decoder(z,training=training)
        x = decoder_out #decoder_out["output"]
        try:
        # Drop the keras mask, so it doesn't scale the losses/metrics.
            del x._keras_mask
        except AttributeError:
            pass
        return {"data":data, "predicted_data":x,"z_mean":z_mean, "z_log_var":z_log_var}

    # def get_reconstruction_probability(self, inputs, num_extracted_samples:int):
    #     data = inputs["inputs"]
    #     reconstruction_vectors = np.zeros([num_extracted_samples, data.shape[1]])
    #     for i in range(num_extracted_samples):
    #         _,reconstruction, z_mean, z_log_var = self(inputs)
    #         # _, _, z = self.encoder(data)
    #         # reconstruction = self.decoder(z)
    #         reconstruction_vectors[i] = reconstruction
        
    def call(self, inputs, training=False):
        return self.__call__(inputs=inputs,training=training)

    # def build(self, input_shape):
    #         self.encoder = Encoder(units=self.units)
    #         self.decoder = Decoder(units=input_shape[-1])
def split_clusters(vae,data,labels):
    z_mean, z_log_var, _ = vae.encoder.predict(data)
    unique_labels = np.unique(labels) # Returns classes in order
    clusters_data = [z_mean[labels==current_label,:] for current_label in unique_labels] # Split cluster data per cluster
    return clusters_data # Return a list, one cluster data per list

def get_vae_clusters_parameters(vae, data, labels):
    # Return mean and std per cluster
    # There are n clusters, where n is the number of distinct labels in label
    # Each sample is the encoded version (latent space) of data trhough the encoder of the vae
    z_mean, z_log_var, _ = vae.encoder.predict(data)
    unique_labels = np.unique(labels)
    cluster_means = np.zeros((unique_labels.shape[0],z_mean.shape[1]))
    for i,current_label in enumerate(unique_labels):
        current_latent_mean = z_mean[labels==current_label,:]
        cluster_means[i] = np.average(current_latent_mean,axis=0)
    return cluster_means


if __name__ == '__main__':
    pass
