import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import BatchNormalization, Conv1D, Dense, Flatten, Conv1DTranspose, \
                        Dropout, InputLayer, AveragePooling1D, Reshape, \
                        UpSampling1D, Masking, Add, SeparableConv1D
import numpy as np
from keras.activations import softplus
from src.utils.nn import bounded_output
from src.utils.fmm import generate_wave_tf, get_A_indexes,get_alpha_indexes, \
                        get_beta_indexes,get_omega_indexes,get_M_indexes, get_wave_indexes, \
                        get_wave_indexes_circular,\
                        get_fmm_num_parameters, get_circular_indexes_as_boolean_t
from src.utils.metrics import masked_mse,mse_timeseries, weighted_mean_squared_error, \
                            circular_weighted_mean_square_error
from src.utils.math import cos_sin_vector_to_angle
from src.utils.fmm import get_A_indexes_circular,get_alpha_indexes_circular,\
                        get_beta_indexes_circular,get_omega_indexes_circular,\
                        get_M_indexes_circular, get_fmm_num_parameters_circular
                        
class FMM_head(tf.keras.layers.Layer):
    def __init__(self, num_leads, seq_len, split_ecg, max_omega, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic)
        self.n = num_leads
        self.num_waves = 5
        self.num_parameters, self.num_parameters_per_wave = get_fmm_num_parameters_circular(num_leads=num_leads,num_waves=self.num_waves)#self.num_parameters_per_wave*self.num_waves + self.n #Add also n parameters for parameter M
        self.dense_layers = [Dense(num_nodes,activation=act) for num_nodes,act in \
                             zip([256,self.num_parameters],["tanh","linear"])]
        self.softplus = softplus
        self.seq_len = seq_len
        self.split_ecg = split_ecg
        self.max_omega = max_omega
        self.coeffs_properties_dict = kwargs.get("coeffs_properties_dict",None)
        if(self.coeffs_properties_dict is not None):
            self.coeffs_properties_dict = {k:tf.expand_dims(tf.convert_to_tensor(v),0) for k,v in self.coeffs_properties_dict.items()}
        self.kwargs = kwargs
        # Order: [nxA,1x2xalpha,nx2xbeta,1xomega]_wave_i where i is {P, Q, R, S, T } in order, then nxM

    def get_A(self, x, wave_index):
        # wave_index is the index in {P, Q, R, S, T }
        start_index,end_index = get_A_indexes_circular(wave_index=wave_index,num_leads=self.n,num_waves=self.num_waves)
        return x[:,start_index:end_index]

    def get_alpha(self, x, wave_index):
        start_index,end_index = get_alpha_indexes_circular(wave_index=wave_index,num_leads=self.n,num_waves=self.num_waves)
        return x[:,start_index:end_index]
    
    def get_beta(self, x, wave_index):
        start_index,end_index = get_beta_indexes_circular(wave_index=wave_index,num_leads=self.n,num_waves=self.num_waves)
        return x[:,start_index:end_index]

    def get_omega(self, x, wave_index):
        start_index,end_index = get_omega_indexes_circular(wave_index=wave_index,num_leads=self.n,num_waves=self.num_waves)
        return x[:,start_index:end_index]

    def get_M(self, x):
        start_index,end_index = get_M_indexes_circular(wave_index=None,num_leads=self.n,num_waves=self.num_waves)
        return x[:,start_index:end_index]
    
    def split_parameters(self,x):
        parameters_dict = {}
        m = self.get_M(x)
        for i,w in enumerate(["P","Q","R","S","T"]):
            a = self.get_A(x,wave_index=i)
            alpha = self.get_alpha(x,wave_index=i)*2-1
            alpha = cos_sin_vector_to_angle(alpha)
            beta = self.get_beta(x,wave_index=i)*2-1
            beta = cos_sin_vector_to_angle(beta)
            omega = self.get_omega(x,wave_index=i)
            parameters_dict[w] = {"A":a,"alpha":alpha,"beta":beta,"omega":omega,"M":m}
        return parameters_dict
    
    def scale_parameters(self, parameters_array,up_alpha,up_beta,up_omega):
        return NotImplementedError()
    
    def bound_parameters(self, parameters_array):
        # return parameters_array
        x = parameters_array
        bounded_parameters_array = []
        for i,w in enumerate(["P","Q","R","S","T"]):
            a = self.get_A(x,wave_index=i)
            a = self.softplus(a)
            alpha = self.get_alpha(x,wave_index=i)
            alpha = bounded_output(alpha,lower=0.0,upper=1.0)
            # alpha = bounded_output(alpha,lower=0,upper=self.upper_limit_alpha_beta)
            beta = self.get_beta(x,wave_index=i)
            beta = bounded_output(beta,lower=0.0,upper=1.0)
            omega = self.get_omega(x,wave_index=i)
            omega = bounded_output(omega,lower=0,upper=self.max_omega)
            # omega = bounded_output(omega,lower=0,upper=1.0)
            bounded_parameters_array.append(a)
            bounded_parameters_array.append(alpha)
            bounded_parameters_array.append(beta)
            bounded_parameters_array.append(omega)
        m = self.get_M(x)
        bounded_parameters_array.append(m)
        bounded_parameters_array = tf.concat(bounded_parameters_array,axis=1)
        # bounded_parameters_array = tf.squeeze(tf.stack(bounded_parameters_array,axis=1))
        return bounded_parameters_array
        
    def get_wave(self, parameters_dict, wave_name, lead, seq_len):
        return generate_wave_tf(parameters_dict=parameters_dict,wave_name=wave_name,
                             lead=lead,seq_len=seq_len,max_len=self.seq_len,split_ecg=self.split_ecg)
    
    def inv_scale_parameters(self,parameters_array_scaled):
        x = parameters_array_scaled
        parameters_array = []
        for i,w in enumerate(["P","Q","R","S","T"]):
            a = self.get_A(x,wave_index=i) 
            alpha = self.get_alpha(x,wave_index=i)*2-1
            alpha = cos_sin_vector_to_angle(alpha)
            beta = self.get_beta(x,wave_index=i)*2-1
            beta = cos_sin_vector_to_angle(beta)
            omega = self.get_omega(x,wave_index=i)
            parameters_array.append(a)
            parameters_array.append(alpha)
            parameters_array.append(beta)
            parameters_array.append(omega)
        m = self.get_M(x)
        parameters_array.append(m)
        # parameters_array = tf.squeeze(tf.stack(parameters_array,axis=1))
        parameters_array = tf.squeeze(tf.concat(parameters_array,axis=1))
        return parameters_array
    
    def call(self, inputs, *args, **kwargs):
        return self.__call__(inputs)
    
    def __call__(self, x, x_len=None,
                 return_parameters_dict=False,
                 return_parameters_array:bool=False,
                 return_parameters_array_scaled:bool=False):
        result = {}
        for layer in self.dense_layers:
            x = layer(x)
        parameters_array_scaled = self.bound_parameters(x)
        parameters_array = self.inv_scale_parameters(parameters_array_scaled)
        parameters_dict = self.split_parameters(parameters_array_scaled)
        if(return_parameters_dict):
            result["parameters_dict"] = parameters_dict
        if(return_parameters_array_scaled):
            result["parameters_array_scaled"] = parameters_array_scaled
        leads = [] 
        for i in range(self.n):
            wave = tf.zeros((tf.shape(parameters_array)[0],self.seq_len))
            if(self.split_ecg):
                seq_len = x_len
            else:
                seq_len = self.seq_len
            for j,w in enumerate(["P","Q","R","S","T"]):
                wave = tf.add(wave, self.get_wave(parameters_dict=parameters_dict,wave_name=w,lead=i,seq_len=seq_len))
            m = parameters_dict["P"]["M"][:,i] #self.get_M(x=parameters_array)#[:,i]
            #while(m.ndim < wave.ndim):
            m = tf.expand_dims(m,-1)
            wave = tf.add(wave, m)
            leads.append(wave)
        tf_leads = tf.stack(leads,axis=-1)
        result["output"] = tf_leads
        if(return_parameters_array):
            result["parameters_array"] = parameters_array
        return result
    
class Base_FMM_AE(tf.keras.Model):
    def __init__(self, num_leads, seq_len, max_omega, batch_size, split_ecg,
                 reconstruction_loss_weight, coefficient_loss_weight,
                 num_warmup_epochs,coeffs_properties_dict, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = None
        self.global_avg_pooling = None
        self.coeffs_properties_dict =  coeffs_properties_dict
        self.split_ecg = split_ecg
        self.num_warmup_epochs = num_warmup_epochs
        self.need_warmup = True if num_warmup_epochs>0 else False
        # self.seq_len = seq_len
        self.encoder_input_type = "tensor" # or "dict" if we feed the whole input dictionary
        self.fmm_head = FMM_head(num_leads=num_leads,seq_len=seq_len,
                                 split_ecg=self.split_ecg,max_omega=max_omega,
                                 batch_size=batch_size,
                                 coeffs_properties_dict=coeffs_properties_dict)
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.coefficient_loss_weight = coefficient_loss_weight
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.coefficient_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.test_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.test_coefficient_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.test_total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.coefficient_loss_tracker,
            self.test_total_loss_tracker,
            self.test_reconstruction_loss_tracker,
            self.test_coefficient_loss_tracker
        ]
    
    def get_encoded_parameters(self,inputs):
        if(self.encoder_input_type=="tensor"):
            x = inputs["inputs"]
            x = tf.expand_dims(x,axis=0)
        elif(self.encoder_input_type=="dict"):
            x = inputs
            x["inputs"] = tf.expand_dims(inputs["inputs"],axis=0)
        x = self.encoder(x)  
        x = self.global_avg_pooling(x)
        for layer in self.fmm_head.dense_layers:
            x = layer(x)
        parameters_dict = self.fmm_head.split_parameters(x)
        return parameters_dict

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss_dict = self.compute_loss(self(inputs=data,training=False))
            total_loss_mean = tf.reduce_mean(loss_dict["total_loss"])
        grads = tape.gradient(total_loss_mean, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss_dict["total_loss"])
        self.reconstruction_loss_tracker.update_state(loss_dict["reconstruction_loss"])
        self.coefficient_loss_tracker.update_state(loss_dict["coefficient_loss"])
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "coefficient_loss": self.coefficient_loss_tracker.result(),
            }
        
    def test_step(self, data):
        loss_dict = self.compute_loss(self(inputs=data,training=False))
        self.test_total_loss_tracker.update_state(loss_dict["total_loss"])
        self.test_reconstruction_loss_tracker.update_state(loss_dict["reconstruction_loss"])
        self.test_coefficient_loss_tracker.update_state(loss_dict["coefficient_loss"])
        return {
            "loss": self.test_total_loss_tracker.result(),
            "reconstruction_loss": self.test_reconstruction_loss_tracker.result(),
            "coefficient_loss": self.test_coefficient_loss_tracker.result(),
            }

    def call(self, inputs, training):
        if(self.encoder_input_type=="tensor"):
            x = inputs["inputs"]
        elif(self.encoder_input_type=="dict"):
            x = inputs
        x = self.encoder(inputs=x,training=training)  # (batch_size, context_len, d_model)
        if(self.global_avg_pooling is not None):
            x = self.global_avg_pooling(x)
        if(self.split_ecg):
            size_without_padding = inputs["sizes"]
        else: 
            size_without_padding = None
        fmm_head_output_dict = self.fmm_head(x,x_len=size_without_padding,return_parameters_array=True,return_parameters_array_scaled=True)
        x = fmm_head_output_dict["output"]
        try:
        # Drop the keras mask, so it doesn't scale the losses/metrics.
            del x._keras_mask
        except AttributeError:
            pass
        if("coefficients" in inputs):
            return {"data":inputs["inputs"], "predicted_data":fmm_head_output_dict["output"], 
                    "fmm_coefficients":inputs["coefficients"], 
                    "fmm_coefficients_ang":inputs["coefficients_ang"], 
                    "predicted_fmm_coefficients":fmm_head_output_dict["parameters_array"],
                    "predicted_fmm_coefficients_scaled":fmm_head_output_dict["parameters_array_scaled"]
                    }
        else:
            return {"data":inputs["inputs"], "predicted_data":fmm_head_output_dict["output"],
                    "predicted_fmm_coefficients":fmm_head_output_dict["parameters_array"],
                    "predicted_fmm_coefficients_scaled":fmm_head_output_dict["parameters_array_scaled"]
                    }

    def compute_loss(self, inputs):
        predicted_data = inputs["predicted_data"]
        data = inputs["data"]
        predicted_fmm_coefficients_scaled = inputs["predicted_fmm_coefficients_scaled"]
        reconstruction_loss = masked_mse(data,predicted_data)
        if("fmm_coefficients" in inputs):
            fmm_coefficients_ang = inputs["fmm_coefficients_ang"]
            num_coefficients = fmm_coefficients_ang.shape[1]
            # Define the weight vector with higher weight for a specific column (R wave)
            weight_factor = 10.0
            weight = tf.ones([num_coefficients])  # All columns have weight 1 initially
            r_indexes = get_wave_indexes_circular(wave_index=2, num_leads=self.fmm_head.n, num_waves=self.fmm_head.num_waves) # R wave is index 2
            weight = tf.tensor_scatter_nd_update(weight, tf.expand_dims(r_indexes, axis=1), weight_factor*tf.ones_like(r_indexes,dtype=float))
            # Higher weight also for M parameter, which is important to have low reconstruction error
            # m_indexes = get_M_indexes_circular(wave_index=2, num_leads=self.fmm_head.n, num_waves=self.fmm_head.num_waves) # Get M indexes
            # weight = tf.tensor_scatter_nd_update(weight, tf.expand_dims(m_indexes, axis=1), weight_factor*tf.ones_like(m_indexes,dtype=float))
            coefficient_loss_dict = weighted_mean_squared_error(x=fmm_coefficients_ang,y=predicted_fmm_coefficients_scaled,w=weight)
            coefficient_loss = coefficient_loss_dict["error"]
        else:
            weight = tf.ones([self.fmm_head.num_parameters])
            coefficient_loss_dict = weighted_mean_squared_error(x=predicted_fmm_coefficients_scaled,y=predicted_fmm_coefficients_scaled,w=weight)
            coefficient_loss = coefficient_loss_dict["error"]
        total_loss =    self.coefficient_loss_weight*(coefficient_loss)+ \
                        self.reconstruction_loss_weight*(reconstruction_loss)                       
        return {"total_loss":total_loss,"reconstruction_loss":reconstruction_loss,
                "coefficient_loss":coefficient_loss,
                "coefficient_loss_vector":coefficient_loss_dict["error_vector"],
                "coefficient_loss_vector_weighted":coefficient_loss_dict["weighted_error_vector"] }

if __name__ == '__main__':
    pass
