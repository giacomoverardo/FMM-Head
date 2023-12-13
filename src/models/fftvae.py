from src.models.vae import *
from scipy.fft import fft,fftfreq, ifft
from src.utils.preprocessing import *
from src.utils.math import *

# import scipy
# fft_data = scipy.fft.fft(data[:,:,1])
# fft_data = np.expand_dims(fft_data,axis=-1)
# data_preprocessed = fft_data
# labels_preprocessed = labels
# num_features = 1
class FFT_VAE(VAE):
    def __init__(self, encoder, decoder, fft_params:dict, alpha=1, beta=1, **kwargs):
        super().__init__(encoder, decoder, alpha, beta, **kwargs)   
        # self.min_val = tf.convert_to_tensor(fft_params["min_val"])
        # self.max_val = tf.convert_to_tensor(fft_params["max_val"])
        self.split_len = fft_params["split_len"]
        self.cut_freq = fft_params["cut_freq"]

    def __call__(self, inputs, training=None, mask=None):
        fft_data = inputs["fft"]
        z_mean, z_log_var, z = self.encoder(fft_data)
        reconstructed_half_fft = self.decoder(z)
        fft_re = reconstructed_half_fft[:,:,0]
        fft_im = reconstructed_half_fft[:,:,1]
        reconstruction = invert_half_fft_tf(fft_re = fft_re,fft_im = fft_im,
                                            min_val =inputs["min_val"],max_val=inputs["max_val"],
                                            split_len=self.split_len, cut_freq=self.cut_freq)
        reconstruction = tf.math.real(reconstruction)
        reconstruction = tf.expand_dims(reconstruction, axis=-1)
        # Add constant component which is deleted in the fft element compared to the inputs
        mean_inputs = tf.math.reduce_mean(inputs["inputs"],axis=1,keepdims=True)
        reconstruction_plus_mean = tf.add(reconstruction,mean_inputs)
        return inputs["inputs"], reconstruction_plus_mean, z_mean, z_log_var
    
    # def compute_loss(self, data, reconstruction, z_mean, z_log_var):
    #     kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    #     kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    #     reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(reconstruction, data),axis=[1,2]),axis=0)
    #     total_loss = tf.reduce_mean(kl_loss + reconstruction_loss)
    #     total_loss = self.alpha*reconstruction_loss + self.beta*kl_loss
    #     return total_loss,reconstruction_loss,kl_loss

def dtw_loss():
    pass
if __name__ == '__main__':
    pass


# atol = 1e-10
# fft_data = scipy.fft.fft(data, axis=1)
# def my_simple_vae(x):
#     return x
# # fft_data_real_without_zero = fft_data.real[:,1:]
# # fft_data_im_without_zero = fft_data.imag[:,1:]
# # s0_re, s1_re = np.array_split(fft_data_real_without_zero, 2, axis=1)
# # s0_im, s1_im = np.array_split(fft_data_im_without_zero, 2, axis=1)
# s0_re,s0_im = get_half_fourier(fft_data)
# s0_abs = s0_re*s0_re + s0_im*s0_im
# split_len = s0_re.shape[1]
# # Cut at frequency
# area_percentage = 0.99
# cut_freq = get_cut_val(s0_abs, area_percentage)
# print(cut_freq)
# # cut_freq = None
# s0_abs_cut = s0_abs[:,:cut_freq]
# s0_re_cut = s0_re[:,:cut_freq]
# s0_im_cut = s0_im[:,:cut_freq]

# #Normalize according to area under abs of Fourier transform
# _,min_val_abs,max_val_abs = normalize_single_rows(s0_abs_cut,return_parameters=True)
# min_val = np.sqrt(min_val_abs)
# max_val = np.sqrt(max_val_abs)
# s0_re_cut_norm = normalize_single_rows(s0_re_cut,min_val=min_val,max_val=max_val)
# s0_im_cut_norm = normalize_single_rows(s0_im_cut,min_val=min_val,max_val=max_val)

# # Apply VAE and get reconstruction
# s0_re_cut_norm_reconstruct = my_simple_vae(s0_re_cut_norm)
# s0_im_cut_norm_reconstruct = my_simple_vae(s0_im_cut_norm)
# # Unnorm
# s0_re_cut_reconstruct = unnormalize_single_rows(s0_re_cut_norm_reconstruct,min_val,max_val)
# s0_im_cut_reconstruct = unnormalize_single_rows(s0_im_cut_norm_reconstruct,min_val,max_val)
# np.testing.assert_allclose(s0_re_cut_reconstruct,s0_re_cut, atol=atol)
# np.testing.assert_allclose(s0_im_cut_reconstruct,s0_im_cut, atol=atol)
# # Pad with zeros (uncut)
# # s0_re_reconstruct = s0_re_cut_reconstruct #np.pad(s0_re_cut_reconstruct,pad_width=[(0,0),(0,split_len-cut_freq)])
# # s0_im_reconstruct = s0_im_cut_reconstruct #np.pad(s0_im_cut_reconstruct,pad_width=[(0,0),(0,split_len-cut_freq)])
# s0_re_reconstruct = np.pad(s0_re_cut_reconstruct,pad_width=[(0,0),(0,split_len-cut_freq)])
# s0_im_reconstruct = np.pad(s0_im_cut_reconstruct,pad_width=[(0,0),(0,split_len-cut_freq)])

# # Recover fft from real and imag part and do ifft
# fft_reconstruct = recombine_half_ffts(s0_re_reconstruct,s0_im_reconstruct)
# reconstruction = scipy.fft.ifft(fft_reconstruct)
# for to_plot in [data,reconstruction]: #[fft_data.real,fft_data.imag,fft_re_reconstruct,fft_im_reconstruct,reconstruction] data,reconstruction
#     print(to_plot.shape)
#     index_to_plot =0
#     # plt.figure()
#     try:
#         plt.plot(to_plot.numpy()[index_to_plot])
#     except:
#         plt.plot(to_plot[index_to_plot,:])