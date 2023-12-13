from src.models.vae import *
from src.utils.preprocessing import *
from src.utils.math import *
from tensorflow import keras

class RythmVAE(keras.Model):
    def __init__(self, vae_morpho, vae_rythm, params, **kwargs):
        super(RythmVAE, self).__init__(**kwargs)
        self.vae_morpho = vae_morpho
        self.vae_rythm = vae_rythm
        self.alpha_morpho = params["alpha_morpho"]
        self.beta_morpho = params["beta_morpho"]
        self.alpha_rythm = params["alpha_rythm"]
        self.beta_rythm = params["beta_rythm"]
        self.alpha_vaes = params["alpha_vaes"]
        self.beta_vaes = params["beta_vaes"]     
        # We have 7 losses:
        # -) Reconstruction loss for morphological/rythmic autoencoder
        # -) KL loss for morphological/rythmic autoencoder
        # -) Reconstruction + kl loss for morphological/rytmic autoencoder
        # -) Final loss which is the average of morpho and rythmic total loss
        self.losses_names =  ["morpho_reconstruction_loss",
                "rythm_reconstruction_loss",
                "morpho_kl_loss",
                "rythm_kl_loss",
                "morpho_loss",
                "rythm_loss",
                "loss"]
        # Reconstruction loss for morphological/rythmic autoencoder
        self.morpho_reconstruction_loss_tracker = keras.metrics.Mean(name="morpho_reconstruction_loss")
        self.rythmic_reconstruction_loss_tracker = keras.metrics.Mean(name="rythm_reconstruction_loss")
        # KL loss for morphological/rythmic autoencoder
        self.morpho_kl_loss_tracker = keras.metrics.Mean(name="morpho_kl_loss")
        self.rythmic_kl_loss_tracker = keras.metrics.Mean(name="rythm_kl_loss")
        # Total loss : Reconstruction + kl loss for morphological/rytmic autoencoder
        self.total_morpho_loss_tracker = keras.metrics.Mean(name="total_morpho_loss")
        self.total_rythm_loss_tracker = keras.metrics.Mean(name="total_rythm_loss")
        # Total loss
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        
        # Test trackers
        # Reconstruction loss for morphological/rythmic autoencoder
        self.test_morpho_reconstruction_loss_tracker = keras.metrics.Mean(name="morpho_reconstruction_loss")
        self.test_rythmic_reconstruction_loss_tracker = keras.metrics.Mean(name="rythm_reconstruction_loss")
        # KL loss for morphological/rythmic autoencoder
        self.test_morpho_kl_loss_tracker = keras.metrics.Mean(name="morpho_kl_loss")
        self.test_rythmic_kl_loss_tracker = keras.metrics.Mean(name="rythm_kl_loss")
        # Total loss : Reconstruction + kl loss for morphological/rytmic autoencoder
        self.test_total_morpho_loss_tracker = keras.metrics.Mean(name="total_morpho_loss")
        self.test_total_rythm_loss_tracker = keras.metrics.Mean(name="total_rythm_loss")
        # Total loss
        self.test_total_loss_tracker = keras.metrics.Mean(name="total_loss")
        
    @property
    def metrics(self):
        return [
            self.morpho_reconstruction_loss_tracker,
            self.rythmic_reconstruction_loss_tracker,
            self.morpho_kl_loss_tracker,
            self.rythmic_kl_loss_tracker,
            self.total_morpho_loss_tracker,
            self.total_rythm_loss_tracker,
            self.total_loss_tracker,
            self.test_morpho_reconstruction_loss_tracker,
            self.test_rythmic_reconstruction_loss_tracker,
            self.test_morpho_kl_loss_tracker,
            self.test_rythmic_kl_loss_tracker,
            self.test_total_morpho_loss_tracker,
            self.test_total_rythm_loss_tracker,
            self.test_total_loss_tracker       
        ]
    
    def compute_losses(self,data):
        [data_morpho,data_rythm], [reconstruction_morpho,reconstruction_rythm], \
            [z_mean_morpho,z_mean_rythm], [z_log_var_morpho,z_log_var_rythm] =  self(data)
        # Compute loss for morpho_vae
        total_morpho_loss,morpho_reconstruction_loss,morpho_kl_loss = \
                        self.compute_loss_vae(data_morpho, reconstruction_morpho, \
                                            z_mean_morpho,z_log_var_morpho, \
                                            self.alpha_morpho,self.beta_morpho)
        #Compute loss for rythm vae
        total_rythm_loss,rythm_reconstruction_loss,rythm_kl_loss = \
            self.compute_loss_vae(data_rythm, reconstruction_rythm, \
                                    z_mean_rythm,z_log_var_rythm, \
                                    self.alpha_rythm,self.beta_rythm)
        total_loss = self.alpha_vaes*total_morpho_loss + self.beta_vaes*total_rythm_loss
        losses =  [morpho_reconstruction_loss,
            rythm_reconstruction_loss,
            morpho_kl_loss,
            rythm_kl_loss,
            total_morpho_loss,
            total_rythm_loss,
            total_loss]
        return losses
        
    def train_step(self, data):
        with tf.GradientTape() as tape:
            losses = self.compute_losses(data)
            # [data_morpho,data_rythm], [reconstruction_morpho,reconstruction_rythm], \
            #     [z_mean_morpho,z_mean_rythm], [z_log_var_morpho,z_log_var_rythm] =  self(data)
            # # Compute loss for morpho_vae
            # total_morpho_loss,morpho_reconstruction_loss,morpho_kl_loss = \
            #              self.compute_loss_vae(data_morpho, reconstruction_morpho, \
            #                                     z_mean_morpho,z_log_var_morpho, \
            #                                     self.alpha_morpho,self.beta_morpho)
            # #Compute loss for rythm vae
            # total_rythm_loss,rythm_reconstruction_loss,rythm_kl_loss = \
            #     self.compute_loss_vae(data_rythm, reconstruction_rythm, \
            #                             z_mean_rythm,z_log_var_rythm, \
            #                             self.alpha_rythm,self.beta_rythm)
            # # Compute total losses by summing
            # total_loss = self.alpha_vaes*total_morpho_loss + self.beta_vaes*total_rythm_loss
            # # total_loss,reconstruction_loss,kl_loss = self.compute_loss(inputs, reconstruction, z_mean, z_log_var)
        grads = tape.gradient(losses[-1], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        dict_out = {}
        for m,up_m,l_name in zip(self.metrics[:7],losses,self.losses_names):
            m.update_state(up_m)
            dict_out[l_name] = m.result()
        return dict_out

    def test_step(self, data):
        losses = self.compute_losses(data)
        dict_out = {}
        for m,up_m,l_name in zip(self.metrics[7:],losses,self.losses_names):
            m.update_state(up_m)
            dict_out[l_name] = m.result()
        return dict_out

    def __call__(self, inputs, training=None, mask=None):
        data_morpho = inputs["data_morpho"]
        data_morpho = tf.reshape(data_morpho,shape=(-1,tf.shape(data_morpho)[-1],1))
        data_rythm = inputs["data_rythm"]
        data_rythm = tf.reshape(data_rythm,shape=(-1,tf.shape(data_rythm)[-1],1))
        z_mean_morpho, z_log_var_morpho, z_morpho = self.vae_morpho.encoder(data_morpho)
        z_mean_rythm , z_log_var_rythm , z_rythm = self.vae_rythm.encoder(data_morpho) # We need to use morpho as input (rythm is the next interval, which is the output)
        reconstruction_morpho = self.vae_morpho.decoder(z_morpho)
        reconstruction_rythm = self.vae_rythm.decoder(z_rythm)
        return [data_morpho,data_rythm], [reconstruction_morpho,reconstruction_rythm], \
                [z_mean_morpho,z_mean_rythm], [z_log_var_morpho,z_log_var_rythm]
        
    def call(self, inputs):
        return self.__call__(inputs)

    def compute_loss_vae(self, data, reconstruction, z_mean, z_log_var,alpha, beta):
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(reconstruction, data),axis=[1,2]),axis=0)
        total_loss = alpha*reconstruction_loss + beta*kl_loss
        return total_loss,reconstruction_loss,kl_loss

if __name__ == '__main__':
    pass