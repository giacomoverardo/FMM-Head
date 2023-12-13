import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import BatchNormalization, Conv1D, Dense, Flatten, Conv1DTranspose, \
                        Dropout, InputLayer, AveragePooling1D, Reshape, \
                        UpSampling1D, Masking, Add, SeparableConv1D
import numpy as np
from keras.activations import softplus
from src.utils.nn import bounded_output
from src.models.transformer  import Bert_ecg
import math
from tensorflow.keras.regularizers import l2
from src.utils.fmm import generate_wave_tf, get_A_indexes,get_alpha_indexes, \
                        get_beta_indexes,get_omega_indexes,get_M_indexes, get_wave_indexes, \
                        get_wave_indexes_circular,\
                        get_fmm_num_parameters, get_circular_indexes_as_boolean_t
from src.models.cvae import Cvae_encoder
from src.utils.metrics import masked_mse,mse_timeseries, weighted_mean_squared_error, \
                            circular_weighted_mean_square_error
from src.utils.math import cos_sin_vector_to_angle
from src.utils.fmm import get_A_indexes_circular,get_alpha_indexes_circular,\
                        get_beta_indexes_circular,get_omega_indexes_circular,\
                        get_M_indexes_circular, get_fmm_num_parameters_circular
                        
# class FMM_head(tf.keras.layers.Layer):
#     def __init__(self, num_leads, seq_len, split_ecg, max_omega, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
#         super().__init__(trainable, name, dtype, dynamic)
#         self.n = num_leads
#         # self.num_parameters_per_wave = (self.n*2 + 2)
#         self.num_waves = 5
#         self.num_parameters, self.num_parameters_per_wave = get_fmm_num_parameters(num_leads=num_leads,num_waves=self.num_waves)#self.num_parameters_per_wave*self.num_waves + self.n #Add also n parameters for parameter M
#         self.dense_layers = [Dense(num_nodes,activation=act) for num_nodes,act in \
#                              zip([256,self.num_parameters],["tanh","linear"])]
#         # self.input_dense = Dense(256,activation='tanh') #,activation='tanh',
#         # self.dense = Dense(self.num_parameters) # 5 waves, each wave [nxA,1xalpha,nxbeta,1xomega] ,kernel_initializer="zeros" ,kernel_regularizer=l2(0.05)
#         self.softplus = softplus
#         self.seq_len = seq_len
#         self.split_ecg = split_ecg
#         self.max_omega = max_omega
#         self.coeffs_properties_dict = kwargs.get("coeffs_properties_dict",None)
#         if(self.coeffs_properties_dict is not None):
#             self.coeffs_properties_dict = {k:tf.expand_dims(tf.convert_to_tensor(v),0) for k,v in self.coeffs_properties_dict.items()}
#         self.kwargs = kwargs
#         self.set_upper_limit_alpha_beta(1.0)
#         # Order: [nxA,1xalpha,nxbeta,1xomega]_wave_i where i is {P, Q, R, S, T } in order, then nxM

#     def set_upper_limit_alpha_beta(self,upper:float):
#         assert isinstance(upper,(float,int))
#         self.upper_limit_alpha_beta = upper
        
#     def get_A(self, x, wave_index):
#         # wave_index is the index in {P, Q, R, S, T }
#         start_index,end_index = get_A_indexes(wave_index=wave_index,num_leads=self.n,num_waves=self.num_waves)
#         return x[:,start_index:end_index]

#     def get_alpha(self, x, wave_index):
#         start_index,end_index = get_alpha_indexes(wave_index=wave_index,num_leads=self.n,num_waves=self.num_waves)
#         return x[:,start_index:end_index]
    
#     def get_beta(self, x, wave_index):
#         start_index,end_index = get_beta_indexes(wave_index=wave_index,num_leads=self.n,num_waves=self.num_waves)
#         return x[:,start_index:end_index]

#     def get_omega(self, x, wave_index):
#         start_index,end_index = get_omega_indexes(wave_index=wave_index,num_leads=self.n,num_waves=self.num_waves)
#         return x[:,start_index:end_index]

#     def get_M(self, x):
#         start_index,end_index = get_M_indexes(wave_index=None,num_leads=self.n,num_waves=self.num_waves)
#         return x[:,start_index:end_index]
    
#     def split_parameters(self,x):
#         parameters_dict = {}
#         m = self.get_M(x)
#         # m = self.softplus(m)
#         for i,w in enumerate(["P","Q","R","S","T"]):
#             a = self.get_A(x,wave_index=i)
#             # a = self.softplus(a)
#             alpha = self.get_alpha(x,wave_index=i)
#             # alpha = bounded_output(alpha,lower=0,upper=2*np.pi)
#             beta = self.get_beta(x,wave_index=i)
#             # beta = bounded_output(beta,lower=0,upper=2*np.pi)
#             omega = self.get_omega(x,wave_index=i)
#             # omega = bounded_output(omega,lower=0,upper=self.max_omega)
#             parameters_dict[w] = {"A":a,"alpha":alpha,"beta":beta,"omega":omega,"M":m}
#         return parameters_dict
    
#     def scale_parameters(self, parameters_array,up_alpha,up_beta,up_omega):
#         #Return scaled version of the parameters
#         # Alpha is divided by 2*pi
#         # Beta is divided by 2*pi
#         # Omega is divided by self.max_omega
#         # TODO: scale also m and A?
#         # return parameters_array
#         x = parameters_array
#         scaled_parameters_array = []
#         for i,w in enumerate(["P","Q","R","S","T"]):
#             a = self.get_A(x,wave_index=i)
            
#             # avg_a = self.get_A(tf.expand_dims(self.coeffs_properties_dict["mean"],0),wave_index=i)
#             # std_a = self.get_A(tf.expand_dims(self.coeffs_properties_dict["std"],0),wave_index=i)
#             # # a = tf.divide((a - avg_a),std_a)
#             # q99_a = self.get_A(tf.expand_dims(self.coeffs_properties_dict["q99"],0))
#             # q_neg_99_a = self.get_A(tf.expand_dims(self.coeffs_properties_dict["q-99"],0))
            
#             # avg_a = self.get_A(self.coeffs_properties_dict["mean"],wave_index=i)
#             # std_a = self.get_A(self.coeffs_properties_dict["std"],wave_index=i)
#             # q99_a = self.get_A(self.coeffs_properties_dict["q99"],wave_index=i)
#             # q_neg_99_a = self.get_A(self.coeffs_properties_dict["q-99"],wave_index=i)
#             # a = tf.divide((a - q_neg_99_a),(q99_a - q_neg_99_a))
            
#             alpha = self.get_alpha(x,wave_index=i)
#             alpha = tf.math.divide(alpha,up_alpha) #self.upper_limit_alpha_beta)
#             beta = self.get_beta(x,wave_index=i)
#             beta = tf.math.divide(beta,up_beta) #self.upper_limit_alpha_beta)
#             omega = self.get_omega(x,wave_index=i)
#             omega = tf.math.divide(omega,up_omega)
#             scaled_parameters_array.append(a)
#             scaled_parameters_array.append(alpha)
#             scaled_parameters_array.append(beta)
#             scaled_parameters_array.append(omega)
#         m = self.get_M(x)
#         # avg_m = self.get_M(tf.expand_dims(self.coeffs_properties_dict["mean"],0))
#         # std_m = self.get_M(tf.expand_dims(self.coeffs_properties_dict["std"],0))
#         # q99_m = self.get_M(tf.expand_dims(self.coeffs_properties_dict["q99"],0))
#         # q_neg_99_m = self.get_M(tf.expand_dims(self.coeffs_properties_dict["q-99"],0))
#         # m = tf.divide((m - avg_m),std_m)
        
#         # avg_m = self.get_M(self.coeffs_properties_dict["mean"])
#         # std_m = self.get_M(self.coeffs_properties_dict["std"])
#         # q99_m = self.get_M(self.coeffs_properties_dict["q99"])
#         # q_neg_99_m = self.get_M(self.coeffs_properties_dict["q-99"])
#         # m = tf.divide((m - q_neg_99_m),(q99_m - q_neg_99_m))
    
#         scaled_parameters_array.append(m)
#         scaled_parameters_array = tf.squeeze(tf.stack(scaled_parameters_array,axis=1))
#         return scaled_parameters_array
    
#     def bound_parameters(self, parameters_array):
#         # return parameters_array
#         x = parameters_array
#         bounded_parameters_array = []
#         for i,w in enumerate(["P","Q","R","S","T"]):
#             a = self.get_A(x,wave_index=i)
#             a = self.softplus(a)
#             alpha = self.get_alpha(x,wave_index=i)
#             # alpha = bounded_output(alpha,lower=0,upper=2*np.pi)
#             alpha = bounded_output(alpha,lower=0,upper=self.upper_limit_alpha_beta)
#             beta = self.get_beta(x,wave_index=i)
#             beta = bounded_output(beta,lower=0,upper=self.upper_limit_alpha_beta)
#             omega = self.get_omega(x,wave_index=i)
#             omega = bounded_output(omega,lower=0,upper=self.max_omega)
#             # omega = bounded_output(omega,lower=0,upper=1.0)
#             bounded_parameters_array.append(a)
#             bounded_parameters_array.append(alpha)
#             bounded_parameters_array.append(beta)
#             bounded_parameters_array.append(omega)
#         m = self.get_M(x)
#         bounded_parameters_array.append(m)
#         bounded_parameters_array = tf.squeeze(tf.stack(bounded_parameters_array,axis=1))
#         return bounded_parameters_array
        
#     def get_wave(self, parameters_dict, wave_name, lead, seq_len):
#         return generate_wave_tf(parameters_dict=parameters_dict,wave_name=wave_name,
#                              lead=lead,seq_len=seq_len,max_len=self.seq_len,split_ecg=self.split_ecg)
#         # a = parameters_dict[wave_name]["A"]
#         # alpha = parameters_dict[wave_name]["alpha"]
#         # beta = parameters_dict[wave_name]["beta"]
#         # omega = parameters_dict[wave_name]["omega"]
#         # if(self.split_ecg==False):    
#         #     wave_i = fmm_wave_tf(A=a[:,lead],alpha=alpha,beta=beta[:,lead],omega=omega,wave_len=seq_len)
#         # elif(self.split_ecg):
#         #     wave_i = fmm_wave_tf_different_lens(A=a[:,lead],alpha=alpha,beta=beta[:,lead],omega=omega,wave_lengths=seq_len,final_len=self.seq_len)
#         # return wave_i
    
#     def inv_scale_parameters(self,parameters_array_scaled):
#         # return parameters_array_scaled
#         x = parameters_array_scaled
#         parameters_array = []
#         for i,w in enumerate(["P","Q","R","S","T"]):
#             a = self.get_A(x,wave_index=i) #TODO scale/rescale A
#             alpha = self.get_alpha(x,wave_index=i)
#             alpha = alpha*((2*np.pi)/self.upper_limit_alpha_beta)
#             # alpha = bounded_output(alpha,lower=0,upper=2*np.pi)
#             # alpha = alpha-0.00001 if alpha>self.upper_limit_alpha_beta else alpha
#             beta = self.get_beta(x,wave_index=i)
#             beta = beta*((2*np.pi)/self.upper_limit_alpha_beta)
#             # beta = beta-0.00001 if beta>self.upper_limit_alpha_beta else beta
#             # if(w=="R"):
#             #     beta = (beta*3) - 1.5 + np.pi # Scale between np.pi+1.5 and np.pi-1.5
#             #     # beta = bounded_output(beta,lower=np.pi-1.5,upper=np.pi+1.5)
#             #     # beta = bounded_output(beta,lower=(np.pi-1.5)/(2*np.pi),upper=(np.pi+1.5)/(2*np.pi))
#             #     # raise ValueError
#             # else:
#             #     # beta = bounded_output(beta,lower=0,upper=2*np.pi)
#             #     beta = beta*2*np.pi
#             omega = self.get_omega(x,wave_index=i)
#             # omega = bounded_output(omega,lower=0,upper=self.max_omega)
#             # omega = omega*self.max_omega
#             parameters_array.append(a)
#             parameters_array.append(alpha)
#             parameters_array.append(beta)
#             parameters_array.append(omega)
#         m = self.get_M(x)
#         avg_m = self.get_M(self.coeffs_properties_dict["mean"])
#         std_m = self.get_M(self.coeffs_properties_dict["std"])
#         q99_m = self.get_M(self.coeffs_properties_dict["q99"])
#         q_neg_99_m = self.get_M(self.coeffs_properties_dict["q-99"])
#         # m = bounded_output(m,lower=q_neg_99_m,upper=q99_m)
#         # m = tf.add(tf.multiply(m,(q99_m - q_neg_99_m)), q_neg_99_m)#tf.divide((m - q_neg_99_m),(q99_m - q_neg_99_m))
#         parameters_array.append(m)
#         parameters_array = tf.squeeze(tf.stack(parameters_array,axis=1))
#         return parameters_array
    
#     def call(self, inputs, *args, **kwargs):
#         return self.__call__(inputs)
    
#     def __call__(self, x, x_len=None,
#                  return_parameters_dict=False,
#                  return_parameters_array:bool=False,
#                  return_parameters_array_scaled:bool=False):
#         result = {}
#         # x = self.input_dense(x)
#         # x = self.dense(x)
#         for layer in self.dense_layers:
#             x = layer(x)
#         parameters_array_scaled = self.bound_parameters(x)
#         parameters_array = self.inv_scale_parameters(parameters_array_scaled)
#         parameters_dict = self.split_parameters(parameters_array)
#         if(return_parameters_dict):
#             result["parameters_dict"] = parameters_dict
#         if(return_parameters_array_scaled):
#             result["parameters_array_scaled"] = parameters_array_scaled
#         leads = [] 
#         for i in range(self.n):
#             wave = tf.zeros((tf.shape(parameters_array)[0],self.seq_len))
#             if(self.split_ecg):
#                 seq_len = x_len
#             else:
#                 seq_len = self.seq_len
#             for j,w in enumerate(["P","Q","R","S","T"]):
#                 wave = tf.add(wave, self.get_wave(parameters_dict=parameters_dict,wave_name=w,lead=i,seq_len=seq_len))
#             # wave += parameters_dict["P"]["M"][i] # add m value for lead i (it's the same for all the waves, so we use P as an example)
#             m = self.get_M(x=parameters_array)#[:,i]
#             wave = tf.add(wave,m)
#             leads.append(wave)
#         tf_leads = tf.stack(leads,axis=-1)
#         result["output"] = tf_leads
#         if(return_parameters_array):
#             result["parameters_array"] = parameters_array
#         return result


# class FMM_AE(Bert_ecg):
#     def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1, **kwargs):
#         super().__init__(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate, **kwargs)
#         self.final_layer = None
#         # self.final_layer.add(tf.keras.layers.GlobalAveragePooling1D())
#         # self.final_layer.add(FMM_head(num_leads=kwargs["num_leads"],seq_len=kwargs["seq_len"]))
#         # self.encoder = Sequential()
#         # self.encoder.add(Dense(256,activation="tanh"))
#         # self.encoder.add(Dense(192,activation="tanh"))
#         # self.encoder.add(Dense(128,activation="tanh"))
#         # self.encoder.add(Dense(96,activation="tanh"))
        
#         # self.encoder.add(Conv1D(128,9,padding="same",activation="tanh"))
#         # self.encoder.add(Conv1D(128,9,padding="same",activation="tanh"))
#         # self.encoder.add(Conv1D(64,9,padding="same",activation="tanh"))
#         # self.encoder.add(Conv1D(64,5,padding="same",activation="tanh"))
#         # self.encoder.add(Conv1D(32,5,padding="same",activation="tanh"))
#         # self.encoder.add(Conv1D(16,5,padding="same",activation="tanh"))
#         self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling1D()
#         self.split_ecg = kwargs["split_ecg"]
#         self.fmm_head = FMM_head(num_leads=kwargs["num_leads"],seq_len=kwargs["seq_len"],split_ecg=self.split_ecg,max_omega=kwargs["max_omega"])
        
#     def get_encoded_parameters(self,inputs):
#         # self.call(inputs)
#         x = inputs["inputs"]
#         x = tf.expand_dims(x,axis=0)
#         x = self.encoder(x)  
#         x = self.global_avg_pooling(x)
#         for layer in self.fmm_head.dense_layers:
#             x = layer(x)
#         # x = self.fmm_head.input_dense(x)
#         # x = self.fmm_head.dense(x)
#         parameters_dict = self.fmm_head.split_parameters(x)
#         return parameters_dict

#     def call(self, inputs):
#         # To use a Keras model with `.fit` you must pass all your inputs in the
#         # first argument.
#         # context, x  = inputs
#         x = inputs["inputs"]
        
#         x = self.encoder(x)  # (batch_size, context_len, d_model)
#         x = self.global_avg_pooling(x)
#         if(self.split_ecg):
#             size_without_padding = inputs["sizes"]
#         else: 
#             size_without_padding = None
#         x = self.fmm_head(x,x_len=size_without_padding)
#         x = x["output"]
#         try:
#         # Drop the keras mask, so it doesn't scale the losses/metrics.
#         # b/250038731
#             del x._keras_mask
#         except AttributeError:
#             pass
#         # Return the final output and the attention weights.
#         return {"data":inputs["inputs"], "predicted_data":x}
        
# class FMM_AE_regression(FMM_AE):        
#     def compute_loss(self, inputs):
#         data = inputs["data"]
#         predicted_data = inputs["predicted_data"]
#         fmm_coefficients = inputs["fmm_coefficients"]
#         predicted_fmm_coefficients = inputs["predicted_fmm_coefficients"]
#         # fmm_coefficients_scaled = self.fmm_head.scale_parameters(fmm_coefficients)
#         # predicted_fmm_coefficients_scaled = self.fmm_head.scale_parameters(predicted_fmm_coefficients)
#         fmm_coefficients_scaled = self.fmm_head.scale_parameters \
#                 (fmm_coefficients,up_alpha=2*np.pi,up_beta=2*np.pi,up_omega=1.0) #self.fmm_head.bound_parameters(fmm_coefficients)
#         predicted_fmm_coefficients_scaled = inputs["predicted_fmm_coefficients_scaled"]
#         # fmm_coefficients_scaled = fmm_coefficients
#         # predicted_fmm_coefficients_scaled = predicted_fmm_coefficients
#         reconstruction_loss = masked_mse(data,predicted_data)
        
#         num_coefficients = fmm_coefficients.shape[1]
#         # Define the weight vector with higher weight for a specific column
#         weight_factor = 100.0
#         weight = tf.ones([num_coefficients])  # All columns have weight 1 initially
#         # start_M_index,end_M_index = get_M_indexes(wave_index=None, 
#         #                              num_leads=self.fmm_head.n,
#         #                              num_waves=self.fmm_head.num_waves)
#         # m_indexes = list(range(start_M_index,end_M_index))  # Index of the column to have 10x weight
#         r_indexes = get_wave_indexes(wave_index=2,num_leads=self.fmm_head.n, num_waves=self.fmm_head.num_waves) # R wave is index 2
#         # weight = tf.tensor_scatter_nd_update(weight, tf.expand_dims(r_index, axis=1), [weight_factor])
#         # weight = tf.tensor_scatter_nd_update(weight, tf.expand_dims(r_indexes, axis=1), weight_factor*tf.ones_like(r_indexes,dtype=float))
#         coefficient_loss = weighted_mean_squared_error(fmm_coefficients_scaled, predicted_fmm_coefficients_scaled, weight)
#         # coefficient_loss = tf.keras.metrics.mean_squared_error(fmm_coefficients_scaled,predicted_fmm_coefficients_scaled)
#         def convert_to_linear(x):
#             return x-np.pi*(tf.math.sign(x-np.pi))
#         predicted_alpha_tf = tf.squeeze(tf.stack([convert_to_linear(self.fmm_head.get_alpha(predicted_fmm_coefficients,wave_index=i))
#                                                   for i in range(self.fmm_head.num_waves)],axis=1))
#         real_alpha_tf = tf.squeeze(tf.stack([convert_to_linear(self.fmm_head.get_alpha(fmm_coefficients,wave_index=i))
#                                              for i in range(self.fmm_head.num_waves)],axis=1))
#         # predicted_alpha_order = tf.argsort(predicted_alpha_tf)
#         # real_alpha_order = tf.argsort(real_alpha_tf)
#         # are_alphas_in_order = tf.cast(tf.reduce_all(tf.equal(predicted_alpha_order,real_alpha_order),axis=-1),dtype=tf.float32) 
#         # total_loss =    self.coefficient_loss_weight*(coefficient_loss)+ \
#         #                 self.reconstruction_loss_weight*(reconstruction_loss+10000*(1.0-are_alphas_in_order))
        
#         def softplus_10(x): return tf.math.log(tf.math.exp(20*x) + 1)
#         regularization_alpha_order = softplus_10(predicted_alpha_tf[:,0]-predicted_alpha_tf[:,1])+ \
#                                         softplus_10(predicted_alpha_tf[:,1]-predicted_alpha_tf[:,2])+ \
#                                         softplus_10(predicted_alpha_tf[:,2]-predicted_alpha_tf[:,3])+ \
#                                         softplus_10(predicted_alpha_tf[:,3]-predicted_alpha_tf[:,4])
#         regularization_alpha_order = 0.0
#         total_loss =    self.coefficient_loss_weight*(coefficient_loss)+ \
#                         self.reconstruction_loss_weight*(reconstruction_loss+10000*regularization_alpha_order)                       
#         # reconstruction_loss = mse_timeseries(data,reconstruction)
#         return {"total_loss":total_loss,"reconstruction_loss":reconstruction_loss,"coefficient_loss":coefficient_loss}
    
    
# class FMM_AE_regression_circular(FMM_AE):

#     def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1, **kwargs):
#         super().__init__(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate, **kwargs)
#         self.coeffs_properties_dict =  kwargs["coeffs_properties_dict"]
#         self.fmm_head = FMM_head(num_leads=kwargs["num_leads"],seq_len=kwargs["seq_len"],
#                                  split_ecg=self.split_ecg,max_omega=kwargs["max_omega"],
#                                  batch_size=kwargs["batch_size"],
#                                  coeffs_properties_dict=kwargs["coeffs_properties_dict"])
#         self.fmm_head.set_upper_limit_alpha_beta(2*np.pi)
#         self.reconstruction_loss_weight = kwargs["reconstruction_loss_weight"]
#         self.coefficient_loss_weight = kwargs["coefficient_loss_weight"]
#         self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
#         self.coefficient_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
#         self.test_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
#         self.test_coefficient_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        
#     @property
#     def metrics(self):
#         return [
#             self.total_loss_tracker,
#             self.reconstruction_loss_tracker,
#             self.coefficient_loss_tracker,
#             self.test_total_loss_tracker,
#             self.test_reconstruction_loss_tracker,
#             self.test_coefficient_loss_tracker
#         ]
        
#     def train_step(self, data):
#         with tf.GradientTape() as tape:
#             loss_dict = self.compute_loss(self(inputs=data,training=False))
#             total_loss_mean = tf.reduce_mean(loss_dict["total_loss"])
#         grads = tape.gradient(total_loss_mean, self.trainable_weights)
#         self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
#         self.total_loss_tracker.update_state(loss_dict["total_loss"])
#         self.reconstruction_loss_tracker.update_state(loss_dict["reconstruction_loss"])
#         self.coefficient_loss_tracker.update_state(loss_dict["coefficient_loss"])
#         return {
#             "loss": self.total_loss_tracker.result(),
#             "reconstruction_loss": self.reconstruction_loss_tracker.result(),
#             "coefficient_loss": self.coefficient_loss_tracker.result(),
#             }
        
#     def test_step(self, data):
#         loss_dict = self.compute_loss(self(inputs=data,training=False))
#         self.test_total_loss_tracker.update_state(loss_dict["total_loss"])
#         self.test_reconstruction_loss_tracker.update_state(loss_dict["reconstruction_loss"])
#         self.test_coefficient_loss_tracker.update_state(loss_dict["coefficient_loss"])
#         return {
#             "loss": self.test_total_loss_tracker.result(),
#             "reconstruction_loss": self.test_reconstruction_loss_tracker.result(),
#             "coefficient_loss": self.test_coefficient_loss_tracker.result(),
#             }

#     def compute_loss(self, inputs):
#         data = inputs["data"]
#         predicted_data = inputs["predicted_data"]
#         fmm_coefficients = inputs["fmm_coefficients"]
#         predicted_fmm_coefficients = inputs["predicted_fmm_coefficients"]
#         # fmm_coefficients_scaled = self.fmm_head.scale_parameters(fmm_coefficients)
#         # predicted_fmm_coefficients_scaled = self.fmm_head.scale_parameters(predicted_fmm_coefficients)
#         fmm_coefficients_scaled = self.fmm_head.scale_parameters \
#                 (fmm_coefficients,up_alpha=1.0,up_beta=1.0,up_omega=1.0)
#         predicted_fmm_coefficients_scaled = inputs["predicted_fmm_coefficients_scaled"]
#         # fmm_coefficients_scaled = fmm_coefficients
#         # predicted_fmm_coefficients_scaled = predicted_fmm_coefficients
#         reconstruction_loss = masked_mse(data,predicted_data)
        
#         num_coefficients = fmm_coefficients.shape[1]
#         # Define the weight vector with higher weight for a specific column
#         weight_factor = 10.0
#         weight = tf.ones([num_coefficients])  # All columns have weight 1 initially
#         r_indexes = get_wave_indexes(wave_index=2,num_leads=self.fmm_head.n, num_waves=self.fmm_head.num_waves) # R wave is index 2
#         # weight = tf.tensor_scatter_nd_update(weight, tf.expand_dims(r_index, axis=1), [weight_factor])
#         weight = tf.tensor_scatter_nd_update(weight, tf.expand_dims(r_indexes, axis=1), weight_factor*tf.ones_like(r_indexes,dtype=float))
#         circular_indexes = get_circular_indexes_as_boolean_t(num_leads=self.fmm_head.n,num_waves=self.fmm_head.num_waves)
#         coefficient_loss_dict = circular_weighted_mean_square_error(x=fmm_coefficients_scaled, 
#                                                                y=predicted_fmm_coefficients_scaled, 
#                                                                w=weight,
#                                                                c=circular_indexes)
#         coefficient_loss = coefficient_loss_dict["error"]
#         # coefficient_loss = tf.keras.metrics.mean_squared_error(fmm_coefficients_scaled,predicted_fmm_coefficients_scaled)
#         total_loss =    self.coefficient_loss_weight*(coefficient_loss)+ \
#                         self.reconstruction_loss_weight*(reconstruction_loss)                       
#         # reconstruction_loss = mse_timeseries(data,reconstruction)
#         return {"total_loss":total_loss,"reconstruction_loss":reconstruction_loss,
#                 "coefficient_loss":coefficient_loss,
#                 "coefficient_loss_vector":coefficient_loss_dict["error_vector"],
#                 "coefficient_loss_vector_weighted":coefficient_loss_dict["weighted_error_vector"] }
    
#     def call(self, inputs, training):
#         x = inputs["inputs"]
#         x = self.encoder(x=x,training=training)  # (batch_size, context_len, d_model)
#         x = self.global_avg_pooling(x)
#         if(self.split_ecg):
#             size_without_padding = inputs["sizes"]
#         else: 
#             size_without_padding = None
#         fmm_head_output_dict = self.fmm_head(x,x_len=size_without_padding,return_parameters_array=True,return_parameters_array_scaled=True)
#         x = fmm_head_output_dict["output"]
#         try:
#         # Drop the keras mask, so it doesn't scale the losses/metrics.
#             del x._keras_mask
#         except AttributeError:
#             pass
#         return {"data":inputs["inputs"], "predicted_data":fmm_head_output_dict["output"], 
#                 "fmm_coefficients":inputs["coefficients"], 
#                 "predicted_fmm_coefficients":fmm_head_output_dict["parameters_array"],
#                 "predicted_fmm_coefficients_scaled":fmm_head_output_dict["parameters_array_scaled"]}

# # class FMM_CAE(FMM_AE_regression_circular):
# #     def __init__(self, input_seq_len, f1,f2, **kwargs):
# #         super(Bert_ecg,self).__init__()
# #         self.kwargs = kwargs
# #         self.global_avg_pooling = tf.keras.layers.Flatten()#tf.keras.layers.GlobalAveragePooling1D()
# #         self.split_ecg = kwargs["split_ecg"]
# #         self.coeffs_properties_dict =  kwargs["coeffs_properties_dict"]
# #         self.fmm_head = FMM_head(num_leads=kwargs["num_leads"],seq_len=kwargs["seq_len"],
# #                                  split_ecg=self.split_ecg,max_omega=kwargs["max_omega"],
# #                                  batch_size=kwargs["batch_size"],
# #                                  coeffs_properties_dict=kwargs["coeffs_properties_dict"])
# #         self.reconstruction_loss_weight = kwargs["reconstruction_loss_weight"]
# #         self.coefficient_loss_weight = kwargs["coefficient_loss_weight"]
# #         # Metrics:
# #         self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
# #         self.test_total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
# #         self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
# #         self.coefficient_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
# #         self.test_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
# #         self.test_coefficient_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
# #         self.fmm_head.set_upper_limit_alpha_beta(2*np.pi)
# #         self.f1 = f1
# #         self.f2 = f2
# #         input_shape = [input_seq_len,kwargs["num_leads"]]
# #         self.encoder = Cvae_encoder( input_shape=input_shape,
# #                                 latent_dim=None, 
# #                                 filter_size_list = [f1, f1, f1, f1, f1, f1, f2, f2, f2], 
# #                                 num_filters_list=  [16, 16, 16, 32, 48, 64, 64, 80, 80], 
# #                                 add_avg_pool_list= [ 1,  1,  1,  1,  1,  0,  0,  0,  0])
# #     def call(self, inputs, training):
# #         x = inputs["inputs"]
# #         x = self.encoder(inputs=x,training=training)["output"]  # (batch_size, context_len, d_model)
# #         x = self.global_avg_pooling(x)
# #         if(self.split_ecg):
# #             size_without_padding = inputs["sizes"]
# #         else: 
# #             size_without_padding = None
# #         fmm_head_output_dict = self.fmm_head(x,x_len=size_without_padding,return_parameters_array=True,return_parameters_array_scaled=True)
# #         x = fmm_head_output_dict["output"]
# #         try:
# #         # Drop the keras mask, so it doesn't scale the losses/metrics.
# #             del x._keras_mask
# #         except AttributeError:
# #             pass
# #         return {"data":inputs["inputs"], "predicted_data":fmm_head_output_dict["output"], 
# #                 "fmm_coefficients":inputs["coefficients"], 
# #                 "predicted_fmm_coefficients":fmm_head_output_dict["parameters_array"],
# #                 "predicted_fmm_coefficients_scaled":fmm_head_output_dict["parameters_array_scaled"]}

# class FMM_head_ang(tf.keras.layers.Layer):
#     def __init__(self, num_leads, seq_len, split_ecg, max_omega, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
#         super().__init__(trainable, name, dtype, dynamic)
#         self.n = num_leads
#         self.num_waves = 5
#         self.num_parameters, self.num_parameters_per_wave = get_fmm_num_parameters_circular(num_leads=num_leads,num_waves=self.num_waves)#self.num_parameters_per_wave*self.num_waves + self.n #Add also n parameters for parameter M
#         self.dense_layers = [Dense(num_nodes,activation=act) for num_nodes,act in \
#                              zip([256,self.num_parameters],["tanh","linear"])]
#         self.softplus = softplus
#         self.seq_len = seq_len
#         self.split_ecg = split_ecg
#         self.max_omega = max_omega
#         self.coeffs_properties_dict = kwargs.get("coeffs_properties_dict",None)
#         if(self.coeffs_properties_dict is not None):
#             self.coeffs_properties_dict = {k:tf.expand_dims(tf.convert_to_tensor(v),0) for k,v in self.coeffs_properties_dict.items()}
#         self.kwargs = kwargs
#         # Order: [nxA,1x2xalpha,nx2xbeta,1xomega]_wave_i where i is {P, Q, R, S, T } in order, then nxM

#     def get_A(self, x, wave_index):
#         # wave_index is the index in {P, Q, R, S, T }
#         start_index,end_index = get_A_indexes_circular(wave_index=wave_index,num_leads=self.n,num_waves=self.num_waves)
#         return x[:,start_index:end_index]

#     def get_alpha(self, x, wave_index):
#         start_index,end_index = get_alpha_indexes_circular(wave_index=wave_index,num_leads=self.n,num_waves=self.num_waves)
#         return x[:,start_index:end_index]
    
#     def get_beta(self, x, wave_index):
#         start_index,end_index = get_beta_indexes_circular(wave_index=wave_index,num_leads=self.n,num_waves=self.num_waves)
#         return x[:,start_index:end_index]

#     def get_omega(self, x, wave_index):
#         start_index,end_index = get_omega_indexes_circular(wave_index=wave_index,num_leads=self.n,num_waves=self.num_waves)
#         return x[:,start_index:end_index]

#     def get_M(self, x):
#         start_index,end_index = get_M_indexes_circular(wave_index=None,num_leads=self.n,num_waves=self.num_waves)
#         return x[:,start_index:end_index]
    
#     def split_parameters(self,x):
#         parameters_dict = {}
#         m = self.get_M(x)
#         for i,w in enumerate(["P","Q","R","S","T"]):
#             a = self.get_A(x,wave_index=i)
#             alpha = self.get_alpha(x,wave_index=i)*2-1
#             alpha = cos_sin_vector_to_angle(alpha)
#             beta = self.get_beta(x,wave_index=i)*2-1
#             beta = cos_sin_vector_to_angle(beta)
#             omega = self.get_omega(x,wave_index=i)
#             parameters_dict[w] = {"A":a,"alpha":alpha,"beta":beta,"omega":omega,"M":m}
#         return parameters_dict
    
#     def scale_parameters(self, parameters_array,up_alpha,up_beta,up_omega):
#         return NotImplementedError()
    
#     def bound_parameters(self, parameters_array):
#         # return parameters_array
#         x = parameters_array
#         bounded_parameters_array = []
#         for i,w in enumerate(["P","Q","R","S","T"]):
#             a = self.get_A(x,wave_index=i)
#             a = self.softplus(a)
#             alpha = self.get_alpha(x,wave_index=i)
#             alpha = bounded_output(alpha,lower=0.0,upper=1.0)
#             # alpha = bounded_output(alpha,lower=0,upper=self.upper_limit_alpha_beta)
#             beta = self.get_beta(x,wave_index=i)
#             beta = bounded_output(beta,lower=0.0,upper=1.0)
#             omega = self.get_omega(x,wave_index=i)
#             omega = bounded_output(omega,lower=0,upper=self.max_omega)
#             # omega = bounded_output(omega,lower=0,upper=1.0)
#             bounded_parameters_array.append(a)
#             bounded_parameters_array.append(alpha)
#             bounded_parameters_array.append(beta)
#             bounded_parameters_array.append(omega)
#         m = self.get_M(x)
#         bounded_parameters_array.append(m)
#         bounded_parameters_array = tf.concat(bounded_parameters_array,axis=1)
#         # bounded_parameters_array = tf.squeeze(tf.stack(bounded_parameters_array,axis=1))
#         return bounded_parameters_array
        
#     def get_wave(self, parameters_dict, wave_name, lead, seq_len):
#         return generate_wave_tf(parameters_dict=parameters_dict,wave_name=wave_name,
#                              lead=lead,seq_len=seq_len,max_len=self.seq_len,split_ecg=self.split_ecg)
    
#     def inv_scale_parameters(self,parameters_array_scaled):
#         x = parameters_array_scaled
#         parameters_array = []
#         for i,w in enumerate(["P","Q","R","S","T"]):
#             a = self.get_A(x,wave_index=i) 
#             alpha = self.get_alpha(x,wave_index=i)*2-1
#             alpha = cos_sin_vector_to_angle(alpha)
#             beta = self.get_beta(x,wave_index=i)*2-1
#             beta = cos_sin_vector_to_angle(beta)
#             omega = self.get_omega(x,wave_index=i)
#             parameters_array.append(a)
#             parameters_array.append(alpha)
#             parameters_array.append(beta)
#             parameters_array.append(omega)
#         m = self.get_M(x)
#         parameters_array.append(m)
#         parameters_array = tf.squeeze(tf.stack(parameters_array,axis=1))
#         return parameters_array
    
#     def call(self, inputs, *args, **kwargs):
#         return self.__call__(inputs)
    
#     def __call__(self, x, x_len=None,
#                  return_parameters_dict=False,
#                  return_parameters_array:bool=False,
#                  return_parameters_array_scaled:bool=False):
#         result = {}
#         for layer in self.dense_layers:
#             x = layer(x)
#         parameters_array_scaled = self.bound_parameters(x)
#         parameters_array = self.inv_scale_parameters(parameters_array_scaled)
#         parameters_dict = self.split_parameters(parameters_array_scaled)
#         if(return_parameters_dict):
#             result["parameters_dict"] = parameters_dict
#         if(return_parameters_array_scaled):
#             result["parameters_array_scaled"] = parameters_array_scaled
#         leads = [] 
#         for i in range(self.n):
#             wave = tf.zeros((tf.shape(parameters_array)[0],self.seq_len))
#             if(self.split_ecg):
#                 seq_len = x_len
#             else:
#                 seq_len = self.seq_len
#             for j,w in enumerate(["P","Q","R","S","T"]):
#                 wave = tf.add(wave, self.get_wave(parameters_dict=parameters_dict,wave_name=w,lead=i,seq_len=seq_len))
#             m = parameters_dict["P"]["M"] #self.get_M(x=parameters_array)#[:,i]
#             wave = tf.add(wave,m)
#             leads.append(wave)
#         tf_leads = tf.stack(leads,axis=-1)
#         result["output"] = tf_leads
#         if(return_parameters_array):
#             result["parameters_array"] = parameters_array
#         return result
    
# class FMM_AE_regression_circular_ang(FMM_AE_regression_circular):

#     def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1, **kwargs):
#         super(FMM_AE_regression_circular,self).__init__(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate, **kwargs)
#         self.coeffs_properties_dict =  kwargs["coeffs_properties_dict"]
#         self.fmm_head = FMM_head_ang(num_leads=kwargs["num_leads"],seq_len=kwargs["seq_len"],
#                                  split_ecg=self.split_ecg,max_omega=kwargs["max_omega"],
#                                  batch_size=kwargs["batch_size"],
#                                  coeffs_properties_dict=kwargs["coeffs_properties_dict"])
#         # self.fmm_head.set_upper_limit_alpha_beta(2*np.pi)
#         self.reconstruction_loss_weight = kwargs["reconstruction_loss_weight"]
#         self.coefficient_loss_weight = kwargs["coefficient_loss_weight"]
#         self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
#         self.coefficient_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
#         self.test_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
#         self.test_coefficient_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        
#     def call(self, inputs, training):
#         return_dict = super().call(inputs=inputs,training=training)
#         return_dict["fmm_coefficients_ang"] = inputs["coefficients_ang"]
#         return return_dict
    
#     def compute_loss(self, inputs):
#         data = inputs["data"]
#         predicted_data = inputs["predicted_data"]
#         # fmm_coefficients = inputs["fmm_coefficients"]
#         fmm_coefficients_ang = inputs["fmm_coefficients_ang"]
        
#         # predicted_fmm_coefficients = inputs["predicted_fmm_coefficients"]
#         # fmm_coefficients_scaled = self.fmm_head.scale_parameters \
#         #         (fmm_coefficients,up_alpha=1.0,up_beta=1.0,up_omega=1.0)
#         predicted_fmm_coefficients_scaled = inputs["predicted_fmm_coefficients_scaled"]
#         reconstruction_loss = masked_mse(data,predicted_data)
#         num_coefficients = fmm_coefficients_ang.shape[1]
#         # Define the weight vector with higher weight for a specific column
#         weight_factor = 10.0
#         weight = tf.ones([num_coefficients])  # All columns have weight 1 initially
#         r_indexes = get_wave_indexes_circular(wave_index=2,num_leads=self.fmm_head.n, num_waves=self.fmm_head.num_waves) # R wave is index 2
#         weight = tf.tensor_scatter_nd_update(weight, tf.expand_dims(r_indexes, axis=1), weight_factor*tf.ones_like(r_indexes,dtype=float))
#         # circular_indexes = get_circular_indexes_as_boolean_t(num_leads=self.fmm_head.n,num_waves=self.fmm_head.num_waves)
#         # coefficient_loss_dict = circular_weighted_mean_square_error(x=fmm_coefficients_scaled, 
#         #                                                        y=predicted_fmm_coefficients_scaled, 
#         #                                                        w=weight,
#         #                                                        c=circular_indexes)
#         coefficient_loss_dict = weighted_mean_squared_error(x=fmm_coefficients_ang,y=predicted_fmm_coefficients_scaled,w=weight)
#         coefficient_loss = coefficient_loss_dict["error"]
#         # coefficient_loss = tf.keras.metrics.mean_squared_error(fmm_coefficients_scaled,predicted_fmm_coefficients_scaled)
#         total_loss =    self.coefficient_loss_weight*(coefficient_loss)+ \
#                         self.reconstruction_loss_weight*(reconstruction_loss)                       
#         # reconstruction_loss = mse_timeseries(data,reconstruction)
#         return {"total_loss":total_loss,"reconstruction_loss":reconstruction_loss,
#                 "coefficient_loss":coefficient_loss,
#                 "coefficient_loss_vector":coefficient_loss_dict["error_vector"],
#                 "coefficient_loss_vector_weighted":coefficient_loss_dict["weighted_error_vector"] }
        

# class FMM_CAE(FMM_AE_regression_circular_ang):
#     def __init__(self, input_seq_len, f1,f2, **kwargs):
#         super(Bert_ecg,self).__init__()
#         self.kwargs = kwargs
#         self.global_avg_pooling = tf.keras.layers.Flatten()#tf.keras.layers.GlobalAveragePooling1D()
#         self.split_ecg = kwargs["split_ecg"]
#         self.coeffs_properties_dict =  kwargs["coeffs_properties_dict"]
#         self.fmm_head = FMM_head_ang(num_leads=kwargs["num_leads"],seq_len=kwargs["seq_len"],
#                                  split_ecg=self.split_ecg,max_omega=kwargs["max_omega"],
#                                  batch_size=kwargs["batch_size"],
#                                  coeffs_properties_dict=kwargs["coeffs_properties_dict"])
#         self.reconstruction_loss_weight = kwargs["reconstruction_loss_weight"]
#         self.coefficient_loss_weight = kwargs["coefficient_loss_weight"]
#         # Metrics:
#         self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
#         self.test_total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
#         self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
#         self.coefficient_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
#         self.test_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
#         self.test_coefficient_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
#         # self.fmm_head.set_upper_limit_alpha_beta(2*np.pi)
#         self.f1 = f1
#         self.f2 = f2
#         input_shape = [input_seq_len,kwargs["num_leads"]]
#         self.encoder = Cvae_encoder( input_shape=input_shape,
#                                 latent_dim=None, 
#                                 filter_size_list = [f1, f1, f1, f1, f1, f1, f2, f2, f2], 
#                                 num_filters_list=  [16, 16, 16, 32, 48, 64, 64, 80, 80], 
#                                 add_avg_pool_list= [ 1,  1,  1,  1,  1,  0,  0,  0,  0])
#     def call(self, inputs, training):
#         x = inputs["inputs"]
#         x = self.encoder(inputs=x,training=training)["output"]  # (batch_size, context_len, d_model)
#         x = self.global_avg_pooling(x)
#         if(self.split_ecg):
#             size_without_padding = inputs["sizes"]
#         else: 
#             size_without_padding = None
#         fmm_head_output_dict = self.fmm_head(x,x_len=size_without_padding,return_parameters_array=True,return_parameters_array_scaled=True)
#         x = fmm_head_output_dict["output"]
#         try:
#         # Drop the keras mask, so it doesn't scale the losses/metrics.
#             del x._keras_mask
#         except AttributeError:
#             pass
#         return {"data":inputs["inputs"], "predicted_data":fmm_head_output_dict["output"], 
#                 "fmm_coefficients":inputs["coefficients"], 
#                 "predicted_fmm_coefficients":fmm_head_output_dict["parameters_array"],
#                 "predicted_fmm_coefficients_scaled":fmm_head_output_dict["parameters_array_scaled"]}


if __name__ == '__main__':
    pass
