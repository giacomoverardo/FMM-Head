import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
from keras.layers import InputSpec, Layer, Dropout, MaxPooling1D, Concatenate, Input, Dense, Reshape, \
    Flatten, Activation, UpSampling1D, Conv1D, Bidirectional, LSTM, LeakyReLU, Masking
from keras.models import Sequential, Model, load_model
from keras import backend as K
from keras import initializers, regularizers, constraints
from typing import List,Dict,Tuple
import sklearn
from src.plot.general import plot_roc
from src.models.baseline import BaselineModel
""" Code from https://github.com/gaofujie1997/ECG-ADGAN/tree/master , which is the provided implementation
    for the paper "A Novel Generative Adversarial Network based Electrocardiography Anomaly Detection".

    Minor non-functional updates have been made to wrap the original code in the pipeline of the project.
"""
class MinibatchDiscrimination(Layer):
    def __init__(self, nb_kernels, kernel_dim, init='glorot_uniform', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        self.init = initializers.get(init)
        self.nb_kernels = nb_kernels
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2

        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
                                 initializer=self.init,
                                 name='kernel',
                                 regularizer=self.W_regularizer,
                                 trainable=True,
                                 constraint=self.W_constraint)

        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, mask=None):
        activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
        return K.concatenate([x, minibatch_features], 1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], input_shape[1] + self.nb_kernels

    def get_config(self):
        config = {'nb_kernels': self.nb_kernels,
                  'kernel_dim': self.kernel_dim,
                  'init': 'GlorotUniform',
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MinibatchDiscrimination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EcgAdGan(BaselineModel):
    def __init__(self, sequence_length:int, num_features:int, latent_size:int, 
                 random_sine:bool, scale:int, minibatch:bool, batch_size:int,
                 save_interval:int, save_model_interval:int, save_model:bool,
                 optimizer, num_warmup_epochs:int, name, **kwargs):
        super().__init__(name=name,sequence_length=sequence_length,num_features=num_features, num_warmup_epochs=num_warmup_epochs)
        self.input_shape = (sequence_length, num_features)
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.random_sine = random_sine
        self.scale = scale
        self.minibatch = minibatch
        # Saving parameters
        self.save_interval = save_interval
        self.save_model_interval = save_model_interval
        self.save_model = save_model
        # Masking layer (not used)
        self.masking = Masking(mask_value=0.0)
        # Create model
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.generator = self.build_generator()
        z = Input(shape=(self.latent_size,))
        signal = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(signal)
        self.combine = Model(z, valid)
        self.combine.compile(loss='binary_crossentropy', optimizer=optimizer)
        
    def build_generator(self):
        model = Sequential(name='Generator_v1')
        model.add(Reshape((self.latent_size, 1)))
        model.add(Bidirectional(LSTM(16, return_sequences=True)))

        model.add(Conv1D(32, kernel_size=8, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling1D())
        model.add(Conv1D(16, kernel_size=8, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling1D())
        model.add(Conv1D(8, kernel_size=8, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv1D(1, kernel_size=8, padding="same"))
        model.add(Flatten())

        model.add(Dense(self.input_shape[0]))
        model.add(Activation('tanh'))
        model.add(Reshape(self.input_shape))
        noise = Input(shape=(self.latent_size,))
        signal = model(noise)

        return Model(inputs=noise, outputs=signal)

    def build_discriminator(self):
        signal = Input(shape=self.input_shape)

        flat = Flatten()(signal)
        mini_disc = MinibatchDiscrimination(10, 3)(flat)

        md = Conv1D(8, kernel_size=8, strides=1, input_shape=self.input_shape, padding='same')(signal)
        md = LeakyReLU(alpha=0.2)(md)
        md = Dropout(0.25)(md)
        md = MaxPooling1D(3)(md)

        md = Conv1D(16, kernel_size=8, strides=1, input_shape=self.input_shape, padding='same')(md)
        md = LeakyReLU(alpha=0.2)(md)
        md = Dropout(0.25)(md)
        md = MaxPooling1D(3, strides=2)(md)

        md = Conv1D(32, kernel_size=8, strides=2, input_shape=self.input_shape, padding='same')(md)
        md = LeakyReLU(alpha=0.2)(md)
        md = Dropout(0.25)(md)
        md = MaxPooling1D(3, strides=2)(md)

        md = Conv1D(64, kernel_size=8, strides=2, input_shape=self.input_shape, padding='same')(md)
        md = LeakyReLU(alpha=0.2)(md)
        md = Dropout(0.25)(md)
        md = MaxPooling1D(3, strides=2)(md)
        md = Flatten()(md)
        concat = Concatenate()([md, mini_disc])
        validity = Dense(1, activation='sigmoid')(concat)

        return Model(inputs=signal, outputs=validity, name="Discriminator")     

    def fit(self, dataset,epochs,validation_data=None,callbacks=None):
        print("ECGADGAN DO NOT SUPPORT FIT. REVERTING TO ORIGINAL IMPLEMENTATION")
        print("CALLBACKS AND VALIDATION DATA WILL NOT BE CONSIDERED")
        # SAVE_INTERVAL = 1000
        # SAVE_MODEL_INTERVAL = 200 # Unused, left only because
        # SAVE_MODEL = True
        checkpoint_callback: tf.keras.callbacks.ModelCheckpoint = callbacks[2]
        self.save_folder = os.path.join(os.path.dirname(checkpoint_callback.filepath),"ecg-adgan")
        self.model_dir, self.image_dir, self.loss_dir = \
            [os.path.join(self.save_folder,subfolder) for subfolder in ["model", "ecg_image", "loss"]]
        self.discr_model_path = os.path.join(self.model_dir,"0_discr.h5")
        self.gen_model_path = os.path.join(self.model_dir,"0_gen.h5")
        X_train = dataset["inputs"]
        [os.makedirs(d,exist_ok=True) for d in [self.model_dir, self.image_dir, self.loss_dir]]
        self.train(epochs, X_train, self.batch_size, self.save_interval, 
                   save=self.save_model, save_model_interval=self.save_model_interval)
        callbacks[4].epoch_times = self.epoch_times
        # Custom object to provide a history-like output
        class Myobject:
            pass
        history = Myobject()
        history.history = self.progress
        return history
        
    def load_weights(self, path):
        folder = os.path.dirname(path)
        self.save_folder = os.path.join(folder,"ecg-adgan")
        self.model_dir, self.image_dir, self.loss_dir = \
            [os.path.join(self.save_folder,subfolder) for subfolder in ["model", "ecg_image", "loss"]]
        files_list = os.listdir(self.model_dir)
        max_index_num = -1
        for file_name in files_list:
            index_num = int(file_name.split("_")[0])
            max_index_num = index_num if max_index_num<index_num else max_index_num
        self.discriminator = load_model(os.path.join(self.model_dir,"%d_discr.h5" % max_index_num),custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination})
        self.generator = load_model(os.path.join(self.model_dir,"%d_gen.h5" % max_index_num))
        return self
    
    def expect_partial(self):
        pass
    
    def compute_roc(self, data_dict, normal_class, filename):
        y_scores = self.discriminator.predict(data_dict["inputs"])
        labels = data_dict["labels"]
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels==normal_class, y_score=y_scores)
        roc_auc = sklearn.metrics.auc(fpr,tpr)
        plot_roc(fpr, tpr, roc_auc, filename)
        roc_dict = { "fpr":fpr.tolist(), "tpr":tpr.tolist(), "thresholds":thresholds.tolist(),"roc_auc":roc_auc}
        return roc_dict
    
    def compute_class_loss(self, data_dict:Dict, classes):
        labels = data_dict["labels"]
        loss = self.discriminator.predict(data_dict["inputs"])
        class_loss_dict = {}
        class_mean_loss = np.zeros_like(classes, dtype=np.float32)
        class_std_loss = np.zeros_like(classes, dtype=np.float32)
        for i,class_name in enumerate(classes):
            loss_class = loss[labels==i]
            class_mean_loss[i] = np.mean(loss_class,axis=0)
            class_std_loss[i] = np.std(loss_class,axis=0)
        class_loss_dict = {"mean":class_mean_loss.tolist(), "std":class_std_loss.tolist()}
        return class_loss_dict

    def get_confusion_matrix(self, data_dict:Dict, threshold:float, normal_class:int):
        labels = data_dict["labels"]
        loss = self.discriminator.predict(data_dict["inputs"])
        real_abnormal_indexes = labels==normal_class
        detected_abnormal_indexes = loss>threshold
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true = real_abnormal_indexes,
                                                    y_pred=detected_abnormal_indexes)
        return confusion_matrix
    
    def get_model_size(self, path: str) -> int:
        discr_path, gen_path = [os.path.join(path,"ecg-adgan","model",name) for name in ["0_discr.h5","0_gen.h5"]]
        model_size_bytes = os.path.getsize(discr_path)
        model_size_bytes += os.path.getsize(gen_path)
        return model_size_bytes
        
    def train(self, epochs, X_train, batch_size=128, save_interval=50, save=False, save_model_interval=100):
        vaild = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        self.progress = {'D_loss': [],
                    'G_loss': [],
                    'acc': []}
        self.epoch_times = []
        flag = 0
        for epoch in range(epochs):
            start_epoch_time = time.time()
            # -------------------
            # Train discriminator
            # -------------------
            # select a random batch of signals
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            signals = X_train[idx]
            # sample noise and generatir a batch of new signals
            noise = self.generate_noise(batch_size, self.random_sine)
            gen_signals = self.generator.predict(noise)
            # train the discriminator (real signals labeled as 1 and fake labeled as 0)
            d_loss_real = self.discriminator.train_on_batch(signals, vaild)
            d_loss_fake = self.discriminator.train_on_batch(gen_signals, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # -------------------
            # Train Generator
            # -------------------
            stop = 1000
            if flag == 0:
                if epoch > stop and (100 * d_loss[1]) > 49.5 and (100 * d_loss[1] < 50.5):
                    self.generator.trainable = False
                    flag = 1
                else:
                    g_loss = self.combine.train_on_batch(noise, vaild)
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
            self.progress['D_loss'].append((d_loss[0]))
            self.progress['acc'].append(d_loss[1])
            self.progress['G_loss'].append((g_loss))
            if epoch % save_interval == 0:
                self.save_image(epoch)
                self.loss_plot(self.progress, path=os.path.join(self.loss_dir,"0.png") )
            if save and epoch > stop:
                if (epoch % save_model_interval == 0 and epoch > 0):
                    self.discriminator.save(self.discr_model_path)
                    self.generator.save(self.gen_model_path)
            end_epoch_time = time.time()
            self.epoch_times.append(end_epoch_time-start_epoch_time)
            
        self.loss_plot(self.progress, path=os.path.join(self.loss_dir,"0.png") )
        self.save_image(epoch)
        self.discriminator.save(self.discr_model_path)
        self.generator.save(self.gen_model_path)

    def save_image(self, epoch):
        plt.ioff()
        r, c = 2, 2
        noise = self.generate_noise(r * c, self.random_sine)
        signals = self.generator.predict(noise) * self.scale
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].plot(signals[cnt])
                cnt += 1
        fig.savefig(os.path.join(self.image_dir,"%d.png" % epoch) )
        plt.close()

    def predict(self, data_dict, *args, **kwargs):
        return self.discriminator.predict(data_dict["inputs"])
    
    def loss_plot(self, hist, path):
        x = range(len(hist['D_loss']))
        y1 = hist['D_loss']
        y2 = hist['G_loss']
        plt.plot(x, y1, label='D_loss')
        plt.plot(x, y2, label='G_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc=1)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        
    def generate_noise(self, batch_size, sinwave=False):
        if sinwave:
            x = np.linspace(-np.pi, np.pi, self.latent_size)
            noise = 0.1 * np.random.random_sample((batch_size, self.latent_size)) + 0.9 * np.sin(x)
        else:
            noise = np.random.normal(0, 1, size=(batch_size, self.latent_size))
        return noise

    @property
    def trainable_weights(self):
        # Collect trainable weights from generator and discriminator
        trainable_weights_list = []
        for w in self.generator.trainable_weights:
            trainable_weights_list.append(w)
        for w in self.discriminator.trainable_weights:
            trainable_weights_list.append(w)
        return trainable_weights_list

    @property
    def non_trainable_weights(self):
        # Collect trainable weights from generator and discriminator
        non_trainable_weights_list = []
        for w in self.generator.non_trainable_weights:
            non_trainable_weights_list.append(w)
        for w in self.discriminator.non_trainable_weights:
            non_trainable_weights_list.append(w)
        return non_trainable_weights_list

if __name__=='__main__':
    pass