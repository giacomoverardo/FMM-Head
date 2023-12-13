import numpy as np
import tensorflow as tf
from src.utils.preprocessing import unnormalize_single_rows_tf
from scipy.stats import pearsonr
from scipy.signal import hilbert
from typing import Tuple, List, Dict

def get_statistics_dict_from_matrix(in_array:np.ndarray)->Dict:
    """Return statistics of matrix rows

    Args:
        in_array (np.ndarray): Input 2d matrix

    Returns:
        Dict: dictionary of statistics
    """
    stats_dict = {}
    for f,n in zip([np.average,np.std,np.max,np.min,
                    lambda x,axis:np.quantile(x,0.95,axis=axis),
                    lambda x,axis:-np.quantile(-x,0.95,axis=axis),
                    lambda x,axis:np.quantile(x,0.99,axis=axis),
                    lambda x,axis:-np.quantile(-x,0.99,axis=axis)],
                ["mean","std","max","min","q95","q-95","q99","q-99"]):
        stats_dict[n] = f(in_array,axis=0).tolist()
    return stats_dict

def get_pairwise_distance(in_matrix:np.ndarray)->np.ndarray:
    """Compute pairwise distance between each row in in_matrix

    Args:
        in_matrix (np.ndarray): Input matrix

    Returns:
        np.ndarray: Matrix containing distances
    """
    a = in_matrix
    b = a.reshape(a.shape[0], 1, a.shape[1])
    distance_matrix = np.sqrt(np.einsum('ijk, ijk->ij', a-b, a-b))
    return distance_matrix

def combine_half_ffts(fft_in_real:np.ndarray, fft_in_imag:np.ndarray):
    # Recombine the real and imag part of a fourier transform of a real signal. No information is lost
    # Additionally, a zero value is added to each row of the fft since that's the fromat of scipy.fft
    def double_fft_from_half(in_mat,flip_coef:int):
        return np.concatenate([in_mat,np.flip(flip_coef*in_mat,axis=1)[:,1:]],axis=1)
    double_fft_real_without_zero = double_fft_from_half(fft_in_real, flip_coef=1.0)
    double_fft_imag_without_zero = double_fft_from_half(fft_in_imag, flip_coef=-1.0)
    def pad_rows_with_initial_zero(in_mat):
        return np.pad(in_mat,pad_width=[(0,0),(1,0)]) #pad row with initial 0
    in_real_pad = pad_rows_with_initial_zero(double_fft_real_without_zero) 
    in_imag_pad = pad_rows_with_initial_zero(double_fft_imag_without_zero)
    # Recover fft from real and imag part
    fft_reconstruct = np.vectorize(complex)(in_real_pad, in_imag_pad)
    return fft_reconstruct

def invert_half_fft_tf(fft_re, fft_im, min_val, max_val,split_len,cut_freq):
    # Unnorm
    fft_re_unnorm = unnormalize_single_rows_tf(fft_re,min_val,max_val)
    fft_im_unnorm = unnormalize_single_rows_tf(fft_im,min_val,max_val)
    # Pad with zeros (uncut)
    fft_re_half = tf.pad(fft_re_unnorm,tf.constant([(0,0),(0,split_len-cut_freq)]),"CONSTANT")
    fft_im_half = tf.pad(fft_im_unnorm,tf.constant([(0,0),(0,split_len-cut_freq)]),"CONSTANT")
    # Recover fft from real and imag part and do ifft
    fft_reconstruct = combine_half_ffts_tf(fft_re_half,fft_im_half)
    reconstruction = tf.signal.ifft(fft_reconstruct)
    return reconstruction

@tf.function
def combine_half_ffts_tf(fft_in_real, fft_in_imag):
    # Recombine the real and imag part of a fourier transform of a real signal. No information is lost
    # Additionally, a zero value is added to each row of the fft since that's the fromat of scipy.fft
    @tf.function
    def double_fft_from_half_tf(in_mat,flip_coef:int):
        inverted_mat = tf.multiply(tf.cast(flip_coef,dtype=in_mat.dtype),in_mat)
        inverted_mat = inverted_mat[:,::-1]
        inverted_mat = inverted_mat[:,1:]
        return tf.concat([in_mat, inverted_mat],axis=1)
        # return np.concatenate([in_mat,np.flip(flip_coef*in_mat,axis=1)[:,1:]],axis=1)
    double_fft_real_without_zero = double_fft_from_half_tf(fft_in_real, flip_coef=1.0)
    double_fft_imag_without_zero = double_fft_from_half_tf(fft_in_imag, flip_coef=-1.0)
    def pad_rows_with_initial_zero(in_mat):
        return tf.pad(in_mat,tf.constant([(0,0),(1,0)]), "CONSTANT") #pad row with initial 0
    in_real_pad = pad_rows_with_initial_zero(double_fft_real_without_zero) 
    in_imag_pad = pad_rows_with_initial_zero(double_fft_imag_without_zero)
    # Recover fft from real and imag part
    fft_reconstruct = tf.dtypes.complex(in_real_pad, in_imag_pad)
    # fft_reconstruct = np.vectorize(complex)(in_real_pad, in_imag_pad)
    return fft_reconstruct


def compute_pearson_c_vectors(x,y):
    #x and y are matrices. Each row is a vector
    # Correlation is computed between elements of the vectors independently
    num_samples,vector_size=x.shape
    np.testing.assert_equal(x.shape,y.shape)
    pearson_vector = np.zeros((vector_size))
    for vector_element_index in range(vector_size):
        x_element_samples = x[:,vector_element_index]
        y_element_samples = y[:,vector_element_index]
        pearson_vector[vector_element_index] = pearsonr(x_element_samples,y_element_samples)[0]
    return pearson_vector

def circ_mean(alpha,axis=None):
    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    return np.arctan2(np.sum(sin_alpha,axis=axis),np.sum(cos_alpha,axis=axis))

def circ_corr_coeff(alpha1:np.ndarray,alpha2:np.ndarray):
    np.testing.assert_array_equal(np.shape(alpha1),np.shape(alpha2),)
    n = np.shape(alpha1)[-1]
    alpha1_bar = np.expand_dims(circ_mean(alpha1,axis=-1),-1)
    alpha2_bar = np.expand_dims(circ_mean(alpha2,axis=-1),-1)
    num = np.sum(np.sin(alpha1 - alpha1_bar) * np.sin(alpha2 - alpha2_bar),axis=-1)
    den = np.sqrt(np.sum(np.power(np.sin(alpha1 - alpha1_bar),2),axis=-1) * np.sum(np.power(np.sin(alpha2 - alpha2_bar),2),axis=-1))
    rho = num / den
    return rho

def mixed_lin_circ_corr_coeff(x,y,c):
    # c is 1 in elements if correspondent element in x y is circular. If zero, variable is linear
    lin_corr_coeffs = compute_pearson_c_vectors(x,y)
    circ_corr_coeffs = circ_corr_coeff(np.transpose(x),np.transpose(y))
    corr_coeffs = c*circ_corr_coeffs + (1-c)*lin_corr_coeffs
    return corr_coeffs

def cos_sin_vector_to_angle(x:tf.Tensor)->tf.Tensor:
    # X is a matrix whose columns are cosine and sine of a variable
    # The function reconstructs the variables from cosine and sine
    # The output is the reconstructed matrix (one vector per row)
    # Each row has half size compared to the original rows
    # Angle is between 0 and 2pi
    num_columns = tf.shape(x)[1]
    # assert num_columns%2==0 # We need an even number (cosine and sine)
    cos_matrix = x[:,::2]
    sin_matrix = x[:,1::2]
    angle_x = tf.math.floormod(tf.math.atan2(sin_matrix,cos_matrix)+2*np.pi,2*np.pi)
    return angle_x

def angle_vector_to_cos_sin(x:np.array,ang_indexes,zero_one_interval=False)->np.array:
    # Convert columns at indexes ang indexes from scalar angular data
    # to 2x vector of cosine and sine
    ang_index_sorted = np.sort(ang_indexes)
    
    num_rows, num_columns = np.shape(x)
    num_angs = np.shape(ang_index_sorted)[0]
    all_index = np.arange(num_columns)
    not_ang_index = np.setdiff1d(all_index,ang_indexes)
    cos_sin_indexes = [[ang_ind+i,ang_ind+i+1] for i,ang_ind in enumerate(ang_index_sorted)]

    cos_sin_indexes = np.concatenate(cos_sin_indexes)
    num_new_columns = num_columns + num_angs
    new_all_index = np.arange(num_new_columns)
    new_not_ang_index = np.setdiff1d(new_all_index,cos_sin_indexes)
    new_mat = np.zeros((num_rows,num_new_columns),dtype=float)
    
    ang_values = x[:,ang_index_sorted]
    cos_matrix = np.cos(ang_values)
    sin_matrix = np.sin(ang_values)
    cos_sin_mat = np.zeros((num_rows,2*num_angs))
    if(zero_one_interval):
        cos_sin_mat[:,::2]=(cos_matrix+1)/2
        cos_sin_mat[:,1::2]=(sin_matrix+1)/2
    else:
        cos_sin_mat[:,::2]=cos_matrix
        cos_sin_mat[:,1::2]=sin_matrix
    new_mat[:,cos_sin_indexes] = cos_sin_mat
    new_mat[:,new_not_ang_index] = x[:,not_ang_index]
    return new_mat
    
def angle_vector_to_cos_sin_vector(x:np.array,ang_indexes)->np.array:
    # Convert columns at indexes ang indexes from scalar angular data
    # to 2x vector of cosine and sine
    ang_index_sorted = np.sort(ang_indexes)
    
    # num_rows, num_columns = np.shape(x)
    num_columns = np.shape(x)[0]
    num_angs = np.shape(ang_index_sorted)[0]
    all_index = np.arange(num_columns)
    not_ang_index = np.setdiff1d(all_index,ang_indexes)
    cos_sin_indexes = [[ang_ind+i,ang_ind+i+1] for i,ang_ind in enumerate(ang_index_sorted)]

    cos_sin_indexes = np.concatenate(cos_sin_indexes)
    num_new_columns = num_columns + num_angs
    new_all_index = np.arange(num_new_columns)
    new_not_ang_index = np.setdiff1d(new_all_index,cos_sin_indexes)
    # new_mat = np.zeros((num_rows,num_new_columns),dtype=float)
    new_vec = np.zeros((num_new_columns),dtype=float)
    
    # ang_values = x[:,ang_index_sorted]
    ang_values = x[ang_index_sorted]
    cos_vec = np.cos(ang_values)
    sin_vec = np.sin(ang_values)
    # cos_sin_mat = np.zeros((num_rows,2*num_angs))
    cos_sin_vec = np.zeros((2*num_angs))
    cos_sin_vec[::2]=cos_vec
    cos_sin_vec[1::2]=sin_vec
    
    new_vec[cos_sin_indexes] = cos_sin_vec
    new_vec[new_not_ang_index] = x[not_ang_index]
    return new_vec
    
# def compute_circcoef_vectors(x,y):
#     #x and y are matrices. Each row is a vector
#     # Correlation is computed between elements of the vectors independently
#     num_samples,vector_size=x.shape
#     np.testing.assert_equal(x.shape,y.shape)
#     pearson_vector = np.zeros((vector_size))
#     for vector_element_index in range(vector_size):
#         x_element_samples = x[:,vector_element_index]
#         y_element_samples = y[:,vector_element_index]
#         pearson_vector[vector_element_index] = pearsonr(x_element_samples,y_element_samples)[0]
#     return pearson_vector

def calculate_avg_std_from_dictionaries(dictionaries):
    # Initialize dictionaries to store the average and standard deviation
    avg_dict = {}
    std_dict = {}
    if len(dictionaries) == 0:
        return avg_dict, std_dict
    # Get all keys from the first dictionary
    keys = dictionaries[0].keys()
    for key in keys:
        # Extract values for the current key from all dictionaries
        values = [d.get(key, 0) for d in dictionaries]
        # Calculate average and standard deviation using numpy
        avg = np.mean(values)
        std = np.std(values)
        # Store the calculated values in the respective dictionaries
        avg_dict[key] = avg
        std_dict[key] = std
    return avg_dict, std_dict

def get_fmm_3d_curve(lead2, leadv2):
    lead2_hilbert = hilbert(lead2)
    lead2_hilbert_re = lead2_hilbert.real
    lead2_hilbert_im = lead2_hilbert.imag
    x = lead2_hilbert_re
    y = lead2_hilbert_im
    z = leadv2 - 2*y
    return x,y,z

if __name__ == '__main__':
    pass