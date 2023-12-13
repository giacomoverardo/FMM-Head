from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
import tensorflow as tf
from typing import List
import scipy
import copy
# from src.utils.math import get_cut_val,get_half_fourier
from sklearn.model_selection import train_test_split
from tensorflow.data import Dataset
import pywt
from typing import List, Dict

def get_cut_val(in_matrix, area_percentage):
    # Return index in in matrix which in average splits each row into areas of percentage area_percentage
    in_dim = in_matrix.ndim
    if(in_dim == 1):
        in_matrix_mean = in_matrix
    elif(in_dim == 2):
        in_matrix_mean = np.mean(in_matrix,axis=0)
    else:
        raise ValueError(f"Input array should have max 2 dimensions, found {in_dim}")
    abs_mean_cum_sum = np.cumsum(in_matrix_mean)
    # print(abs_mean_cum_sum.shape)
    cut_val = np.searchsorted(abs_mean_cum_sum, np.max(abs_mean_cum_sum)*area_percentage)
    if(cut_val>=in_matrix.shape[-1]):
        return None
    return cut_val

def get_half_fourier(in_fourier_matrix:np.ndarray):
    # Half the real and imag part of a fourier transform of a real signal. No information is lost
    fft_data_real_without_zero = in_fourier_matrix.real[:,1:] # Delete constant component
    fft_data_im_without_zero = in_fourier_matrix.imag[:,1:]
    s0_re, _ = np.array_split(fft_data_real_without_zero, 2, axis=1)
    s0_im, _ = np.array_split(fft_data_im_without_zero, 2, axis=1)
    # s0_abs = s0_re*s0_re + s0_im*s0_im
    return s0_re,s0_im

def normalize_single_rows(in_matrix,return_parameters:bool=False,min_val=None, max_val=None):
    if(min_val is None):
        min_val = np.min(in_matrix,axis=1,keepdims=True)
    if(max_val is None):
        max_val = np.max(in_matrix,axis=1,keepdims=True)
    # max_val_abs = np.max(np.abs(in_matrix),axis=1,keepdims=True)
    # max_val = max_val_abs
    # min_val = np.zeros_like(min_val)
    # out_matrix = (in_matrix - min_val) / (max_val-min_val)
    out_matrix = 2*(in_matrix - min_val) / (max_val-min_val)-1 # To normalize between -1 and 1
    if(return_parameters):
        return out_matrix, min_val, max_val
    else:
        return out_matrix

def unnormalize_single_rows(norm_matrix, min_val, max_val):
    out_matrix = (norm_matrix*(max_val-min_val)) + min_val
    return out_matrix

@tf.function
def unnormalize_single_rows_tf(norm_matrix, min_val, max_val):
    return tf.math.multiply(norm_matrix, (max_val-min_val)) + tf.convert_to_tensor(min_val)
    
def subtract_mean_from_single_rows(in_matrix,return_parameters:bool=False,mean_val:float=None):
    if(mean_val is None):
        mean_val = np.mean(in_matrix,axis=1,keepdims=True)
    out_matrix = in_matrix - mean_val
    if(return_parameters):
        return out_matrix, mean_val
    else:
        return out_matrix

    # scaler = MinMaxScaler() #StandardScaler()
    # scaler.fit(np.transpose(inArray))
    # outArray = np.transpose(scaler.transform(np.transpose(inArray)))
    # outArray = scaler.fit_transform(inArray)
    # scaler=None
    # outArray = (inArray - np.min(inArray,axis=1))/(np.max(inArray,axis=1)
    # index=10
    # plot_series_list([inArray[index],outArray[index]],legend=["Original","Scaled"])
    # print(scaler.data_max_.shape)

def get_peaks_indexes_for_ecg(in_matrix:np.ndarray,minimum_height:float=0.2,peak_to_peak_distance:int=50):
    # matrix_dim = in_matrix.shape
    # out_peaks = np.zeros(matrix_dim[:2],dtype=bool)
    peaks_indexes = []
    
    for i,vector in enumerate(in_matrix):
        peaks,_ = scipy.signal.find_peaks(np.squeeze(vector), height=minimum_height, 
                                          threshold=None, distance=peak_to_peak_distance, 
                                          prominence=None, width=None, wlen=None, 
                                          rel_height=None, plateau_size=None)
        # out_peaks[i,peaks]= 1
        peaks_indexes.append(peaks)
    return peaks_indexes

def get_peaks_indexes_wavelet(in_matrix:np.ndarray,minimum_height:float=0.2,peak_to_peak_distance:int=50):
    # Apply wavelet filtering (only 3-4 coeffs are kept, the others are zeroed out)
    w = pywt.Wavelet('sym4')
    wt_coeffs = pywt.wavedec(in_matrix,wavelet=w,level=5,axis=1)
    new_wt_coeffs = [ c if (i==3 or i==4) else np.zeros_like(c) for i,c in enumerate(wt_coeffs)]
    # def threshold_abs_function(in_vec, thresh):
    #     out_vec= np.zeros_like(in_vec)
    #     indexes = np.abs(in_vec)>=thresh
    #     out_vec[indexes] = in_vec[indexes]
    #     return out_vec
    # new_wt_coeffs = [ threshold_abs_function(c,thresh = 0.2) for c in wt_coeffs]
    wavelet_in_matrix = pywt.waverec(coeffs=new_wt_coeffs,wavelet=w,axis=1)  
    wavelet_in_matrix_sq = np.square(wavelet_in_matrix)
    return get_peaks_indexes_for_ecg(wavelet_in_matrix_sq,minimum_height,peak_to_peak_distance)

def split_ecg_in_waves(in_matrix:np.ndarray, peaks_indexes:List):
    #TODO: divide it in parts not depending on simple peak indexes, but by putting the split value 
    # so that the peak of each shape is pretty much in the middle of the shape
    sections_list = []
    for ecg, ecg_peaks in zip(in_matrix,peaks_indexes):
        sections = np.split(ecg,ecg_peaks)
        sections_list.append(sections)
    return sections_list

def delete_sequences_longer_than(in_sections_list:List, in_labels:np.ndarray,threshold:int):
    filtered_section_list = []
    filtered_labels_list = []
    for i,sections in enumerate(in_sections_list):
        filtered_sections = []
        filtered_labels = []
        for j,section in enumerate(sections):
            section_size = np.shape(section)[0]
            if(section_size<=threshold): 
                filtered_sections.append(section)
                # try:
                filtered_labels.append(in_labels[i][j])
                # except:
                #     a=0
        filtered_section_list.append(filtered_sections)
        filtered_labels_list.append(filtered_labels)
    return filtered_section_list, filtered_labels_list
                
# def split_ecg_around_peaks(in_matrix:np.ndarray, peaks_indexes:List, low:int, high:int):
#     sections_list = []
#     for ecg, ecg_peaks in zip(in_matrix,peaks_indexes):
#         num_sections = len(peaks_indexes)
#         sections = np.zeros((num_sections,high+low+1))
#         for i,peak in enumerate(ecg_peaks):
#             sections[i,:] = ecg[peak-low,peak+high]
#         sections_list.append(sections)
#     return sections_list

def remove_incorrectly_divided_samples(in_sections_list:List, in_labels:np.ndarray, min_ecg_time:int=None, max_ecg_time:int=None, min_num_sections:int=3):
    # Remove sequences with less than min_num_sections sections (including first and last, which are usually incomplete)
    # and more than max_num_sections. 
    # Also, remove sequences that have sections which have less than min_ecg_time or more than max_ecg_time 
    print(f"Keeping sequences with more than {min_num_sections} sections")
    filtered_section_list = []
    filtered_labels = []
    correct_indices = []
    section_sizes_list = []
    for i,sections in enumerate(in_sections_list):
        section_ok = True
        num_sections = len(sections)
        if(num_sections>min_num_sections):
            section_sizes = np.zeros(num_sections)
            for k,s in enumerate(sections):
                if(k==0 or k==(num_sections-1)):
                    continue # Ignore first and last section
                section_sizes[k] = s.shape[0]
                # s_size = s.shape[0]
                # if(s_size>max_ecg_time or s_size<=min_ecg_time):
                #     section_ok = False
                #     break
            middle_section_sizes = section_sizes[1:-1]
            #Check that there are not outliers sizes (incorrentry segmented ecg)
            sizes_mean = np.mean(middle_section_sizes)
            sizes_std = np.std(middle_section_sizes)
            b = 3
            section_sizes_ok = np.logical_and(middle_section_sizes<=sizes_mean+b*sizes_std,middle_section_sizes>sizes_mean-b*sizes_std)
            if(max_ecg_time):
                section_sizes_ok = np.logical_and(middle_section_sizes<=max_ecg_time,section_sizes_ok)
            if(min_ecg_time):
                section_sizes_ok = np.logical_and(middle_section_sizes>min_ecg_time,section_sizes_ok)
            section_ok = np.all(section_sizes_ok)
            # if(section_sizes_ok.all()==False):
            #     section_ok = False
                # if(s_size>max_ecg_time or s_size<=min_ecg_time):
                #     section_ok = False
                #     break
        else:
            section_ok=False
        if(section_ok):
            filtered_section_list.append(sections)
            filtered_labels.append(in_labels[i])
            section_sizes_list.append(middle_section_sizes)
            correct_indices.append(i)
    # for i,sections in enumerate(filtered_section_list):
    #     for s in sections:
    #         s_size = s.shape[0]
    #         assert s_size < max_ecg_time
    filtered_labels = np.array(filtered_labels)
    correct_indices = np.array(correct_indices)   
    return filtered_section_list, filtered_labels, correct_indices,section_sizes_list

def pad_sequences(in_sections_list:List,  pad_len:int, center:bool=False):
    padded_section_list = []
    num_features = in_sections_list[0][0].shape[-1]
    for i,sections in enumerate(in_sections_list):
        padded_section = np.zeros(shape=(len(sections),pad_len,num_features))
        # padded_section = np.zeros(shape=(pad_len,len(sections)))
        for k,s in enumerate(sections):
            s_size = np.shape(s)[0]
            padded_ecg = np.pad(s,pad_width=((0,pad_len-s_size),(0,0)))
            if(center):
                padded_ecg = np.roll(padded_ecg, int((pad_len-s_size)/2))
            padded_section[k]= padded_ecg
            # padded_section[:,k]= padded_ecg
        padded_section_list.append(padded_section)
    return padded_section_list

def reduce_to_batch(list_data, in_batch_size):
    # Remove samples that do no fit in batch
    out_list = []
    num_0_samples = list_data[0].shape[0]
    num_batched_samples = int(num_0_samples/in_batch_size)*in_batch_size
    for x in list_data:
        num_x_samples = x.shape[0]
        assert num_x_samples == num_0_samples
        new_x = x[:num_batched_samples]
        out_list.append(new_x)
    return out_list

def preprocess_data(in_data,in_labels, classes, params, get_val_set=False):
    fs = params["fs"]
    split_ecg = params["split_ecg"]
    cut_area_percentage = params["cut_area_percentage"]
    batch_size = params["batch_size"]
    undersample = params["undersample"]
    # rythm = params["rythm"]
    data_preprocessed = copy.deepcopy(in_data.astype(np.float32))
    labels_preprocessed = copy.deepcopy(in_labels)
    sos = scipy.signal.butter(5, [0.5,30], btype="bandpass", fs=fs, output='sos')
    data_preprocessed = scipy.signal.sosfilt(sos, data_preprocessed,axis=1).astype(np.float32)
    # # Apply wavelet filtering (only 3-4 coeffs are kept, the others are zeroed out)
    # w = pywt.Wavelet('sym4')
    # wt_coeffs = pywt.wavedec(data_preprocessed,wavelet=w,level=5,axis=1)
    # new_wt_coeffs = [ c if (i==3 or i==4) else np.zeros_like(c) for i,c in enumerate(wt_coeffs)]
    # def threshold_abs_function(in_vec, thresh):
    #     out_vec= np.zeros_like(in_vec)
    #     indexes = np.abs(in_vec)>=thresh
    #     out_vec[indexes] = in_vec[indexes]
    #     return out_vec
    # # new_wt_coeffs = [ threshold_abs_function(c,thresh = 0.2) for c in wt_coeffs]
    # data_preprocessed = pywt.waverec(coeffs=new_wt_coeffs,wavelet=w,axis=1)  
    # np.testing.assert_allclose(data_preprocessed,data_preprocessed2)
    data_preprocessed = normalize_single_rows(data_preprocessed).astype(np.float32)
    #Resample to 250 Hz (2.5x)
    # data_preprocessed =scipy.signal.resample(data_preprocessed,2500,axis=1)
    # Apply z score normalization
    # data_preprocessed = (data_preprocessed-np.mean(data_preprocessed,axis=0,keepdims=True))/np.std(data_preprocessed,axis=0,keepdims=True)
    # Optional: cut to 2048 samples
    # data_preprocessed = scipy.signal.medfilt(data_preprocessed, kernel_size=(1,31))
    #TODO maybe use as usual in_data instead of data_preprocessed
    sequence_length = int(1.6*fs) 
    peaks_indexes = get_peaks_indexes_wavelet(in_data,minimum_height=0.01, peak_to_peak_distance=int(0.4*fs))
    sections_list = split_ecg_in_waves(in_matrix = in_data,peaks_indexes=peaks_indexes)
    # Delete samples which have sections with too small or too high sizes
    min_num_sections = 5
    filtered_section_list, filtered_labels, correct_indices, seq_sizes = remove_incorrectly_divided_samples(sections_list,in_labels,min_ecg_time=None, #int(0.7*fs)
                                                                                max_ecg_time=sequence_length, min_num_sections=2+min_num_sections)
    print("Supposed correctly segmented ecg: {}".format(correct_indices.shape[0]))
    # Delete first and last section for each sample
    for sections in filtered_section_list:
        sections.pop(0)
        sections.pop(-1)
    #TODO: center sections around R peak
    # new_filtered_section_list = []
    # for sections in filtered_section_list:
    #     new_sections = []
    #     for s in sections:   
    #         new_s = 
    #     new_filtered_section_list.append(new_sections)    
    # Pad sequences (optional: center them after padding)
    padded_section_list = pad_sequences(filtered_section_list,sequence_length, center=False)
    num_sections_per_sample = [sections.shape[0] for sections in padded_section_list]
    max_num_sections = np.max(num_sections_per_sample)
    # padded_next_section_list = [np.roll(sections,shift=-1,axis=0) for sections in padded_section_list]
    padded_padded_section_list= []
    # Add zero sections to get to the correct number of sections
    for i,(sections,num_sections) in enumerate(zip(padded_section_list,num_sections_per_sample)):
        padded_with_zero_sections = np.zeros((max_num_sections,sections.shape[1]))
        padded_with_zero_sections[:num_sections,:] = sections
        # for i in range(max_num_sections):
        #     if(i<num_sections):
        #         padded_with_zero_sections.append(sections[i])
        #     else:
        #         padded_with_zero_sections.append(np.zeros_like(sections[num_sections-1]))
        padded_padded_section_list.append(padded_with_zero_sections)
    padded_section_list = padded_padded_section_list 
    num_sections_per_sample = [sections.shape[0] for sections in padded_section_list]
    for x in num_sections_per_sample:
        assert x==max_num_sections
            # elif(i==num_sections):
    rythm_in = padded_section_list      
    # rythm_in = [sections[:min_num_sections-1] for sections in padded_section_list]
    rythm_out = padded_section_list #[sections[:min_num_sections-1] for sections in padded_next_section_list]
    # Delete to avoid morphological-only anomaly detection
    if(split_ecg):
        data_preprocessed = np.concatenate(padded_section_list,axis=0).astype(np.float32)
        #Apply wavelet filtering (only 3-4 coeffs are kept, the others are zeroed out)
        # w = pywt.Wavelet('sym4')
        # wt_coeffs = pywt.wavedec(input_data,wavelet=w,level=5,axis=1)
        # new_wt_coeffs = [ c if (i==2 or i==3) else np.zeros_like(c) for i,c in enumerate(wt_coeffs)]
        # data_preprocessed = pywt.waverec(coeffs=new_wt_coeffs,wavelet=w,axis=1)  
        labels_preprocessed = np.repeat(filtered_labels,repeats=num_sections_per_sample)
        
    # print(np.sum(np.mean(data_preprocessed,axis=1)),np.mean(data,axis=1).shape)
    # data_preprocessed = subtract_mean_from_single_rows(data_preprocessed)
    # print(np.sum(np.mean(data_preprocessed,axis=1)),np.mean(data,axis=1).shape)
    # data_preprocessed = normalize_single_rows(data_preprocessed).astype(np.float32)
    # print(np.sum(np.mean(data_preprocessed,axis=1)),np.mean(data,axis=1).shape)
    # data_preprocessed = data_preprocessed.astype(np.float32)

    # num_samples_per_class = [np.sum(labels_preprocessed==i) for i in range(len(classes))]
    # print(f"Number of samples per class: {num_samples_per_class}")
    if(undersample): #TODO: this does not make sense when rythm is used because the oepration is done on whole dataset (not only correct indexes). Change it
        num_reduced_samples_per_class = np.min(num_samples_per_class)
        print((f"Keeping only {num_reduced_samples_per_class} samples per class"))
        data_preprocessed_reduced = np.zeros((int(len(classes)*num_reduced_samples_per_class),data_preprocessed.shape[1]))
        labels_preprocessed_reduced = np.zeros((int(len(classes)*num_reduced_samples_per_class),))
        for i,_ in enumerate(classes):
            class_indexes = np.where(labels_preprocessed==i)
            class_data = data_preprocessed[class_indexes]
            class_labels = labels_preprocessed[class_indexes]
            reduced_class_data = class_data[:num_reduced_samples_per_class]#TODO: add random selection instead of first elements
            reduced_class_labels = class_labels[:num_reduced_samples_per_class]
            np.testing.assert_allclose(reduced_class_labels,i*np.ones_like(reduced_class_labels))
            first_el_index = int(i*num_reduced_samples_per_class)
            last_el_index = int((i+1)*num_reduced_samples_per_class)
            # print(first_el_index,last_el_index)
            data_preprocessed_reduced[first_el_index:last_el_index,:]=\
                reduced_class_data
            labels_preprocessed_reduced[first_el_index:last_el_index]=\
                reduced_class_labels
        # print(labels_preprocessed_reduced.shape)
        shuffle_perm = np.random.permutation(labels_preprocessed_reduced.shape[0])
        data_preprocessed, labels_preprocessed = data_preprocessed_reduced[shuffle_perm], labels_preprocessed_reduced[shuffle_perm]
        
    while(data_preprocessed.ndim<3): # Expand dimensions until we get 3 dimensions (sample, )
        data_preprocessed = np.expand_dims(data_preprocessed, axis=-1).astype(np.float32)

    fft_data = scipy.fft.fft(data_preprocessed[:,:,0],axis=1)
    s0_re,s0_im = get_half_fourier(fft_data)
    s0_abs = s0_re*s0_re + s0_im*s0_im
    split_len = s0_re.shape[1]

    # Cut at frequency
    cut_freq = get_cut_val(s0_abs, cut_area_percentage)
    print(f"Cut_freq index for fft: {cut_freq}")
    s0_abs_cut = s0_abs[:,:cut_freq]
    s0_re_cut = s0_re[:,:cut_freq]
    s0_im_cut = s0_im[:,:cut_freq]

    #Normalize according to area under abs of Fourier transform
    _,min_val_abs,max_val_abs = normalize_single_rows(s0_abs_cut,return_parameters=True)
    min_val = np.sqrt(min_val_abs)
    max_val = np.sqrt(max_val_abs)
    s0_re_cut_norm = normalize_single_rows(s0_re_cut,min_val=min_val,max_val=max_val)
    s0_im_cut_norm = normalize_single_rows(s0_im_cut,min_val=min_val,max_val=max_val)
    fft_data_preprocessed = np.stack([s0_re_cut_norm,s0_im_cut_norm],axis=-1)
    fft_params = {"split_len":split_len,"cut_freq":cut_freq}
    
    if(get_val_set):      
        # TODO Add the rythm components also for get_val_set==True
        train_data, val_data, train_labels, val_labels, train_fft,val_fft,train_min_val,val_min_val,train_max_val,val_max_val, = \
            train_test_split(data_preprocessed,labels_preprocessed,fft_data_preprocessed,min_val,max_val,test_size=0.2)
        train_data,train_labels,train_fft,train_min_val,train_max_val = reduce_to_batch([train_data,train_labels,train_fft,train_min_val,train_max_val],in_batch_size=batch_size)
        val_data,val_labels,val_fft,val_min_val,val_max_val = reduce_to_batch([val_data,val_labels,val_fft,val_min_val,val_max_val],in_batch_size=batch_size)
        train_dict = {"inputs":train_data,"labels":train_labels,"fft":train_fft,"min_val":train_min_val,"max_val":train_max_val}
        val_dict = {"inputs":val_data,"labels":val_labels,"fft":val_fft,"min_val":val_min_val,"max_val":val_max_val}
        # test_dict = {"inputs":train_data,"labels":train_labels,"fft":train_fft,"min_val":train_min_val,"max_val":train_max_val}
        train_dataset = Dataset.from_tensor_slices(train_dict).shuffle(5000).batch(batch_size,drop_remainder=True)
        val_dataset = Dataset.from_tensor_slices(val_dict).batch(batch_size,drop_remainder=True)
        return train_dict,train_dataset,val_dict,val_dataset,fft_params
    else:
        # Shuffle data before deleting elements to fit batch size
        # data_preprocessed, _, labels_preprocessed, _, fft_data_preprocessed,_,min_val,_,max_val,_, = \
        rythm_out = np.array(rythm_out,dtype=np.float32)
        rythm_in = np.array(rythm_in,dtype=np.float32)
        # correct_indices = correct_indices[correct_indices<data_preprocessed.shape[0]]
        data_preprocessed = data_preprocessed[correct_indices].astype(np.float32)
        labels_preprocessed = labels_preprocessed[correct_indices].astype(np.float32)
        fft_data_preprocessed = fft_data_preprocessed[correct_indices].astype(np.float32)
        min_val = min_val[correct_indices].astype(np.float32)
        max_val = max_val[correct_indices].astype(np.float32)
        data_preprocessed,labels_preprocessed,fft_data_preprocessed,min_val,max_val,rythm_out,rythm_in = \
            reduce_to_batch([data_preprocessed,labels_preprocessed,fft_data_preprocessed,min_val,max_val,rythm_out,rythm_in],in_batch_size=batch_size)
        # data_dict = {"inputs":data_preprocessed,"labels":labels_preprocessed,"fft":fft_data_preprocessed,"min_val":min_val,"max_val":max_val}
        # data_dataset = Dataset.from_tensor_slices(data_dict).batch(batch_size,drop_remainder=True)
        num_samples_per_class = [np.sum(labels_preprocessed==i) for i in range(len(classes))]
        print(f"Number of samples per class: {num_samples_per_class}")
        data_dict = {"inputs":data_preprocessed,"labels":labels_preprocessed,"fft":fft_data_preprocessed,
                    "min_val":min_val,"max_val":max_val,
                    "data_morpho":rythm_in, "data_rythm":rythm_out}       
        data_dataset = Dataset.from_tensor_slices(data_dict).batch(batch_size,drop_remainder=True) 
        return data_dict,data_dataset,fft_params
    
def get_only_normal_train_val_dataset(in_dict:Dict, normal_class:int, batch_size:int, 
                                      seed:int, only_normal:bool=True, val_size:float=0.2,
                                      return_type="dataset"):
    # Return train and val tensorflow dataset taken from in_dict.
    # Only normal samples are returned. Normal samples are the ones with label equal to normal class
    if(only_normal):
        normal_indexes = in_dict["labels"]==normal_class
    else:
        normal_indexes = in_dict["labels"]>-1
    out_dict = {}
    values_list = []
    for key,value in in_dict.items():
        out_dict[key] = value[normal_indexes]
        values_list.append(out_dict[key])
    if(val_size>0.0):
        splitted_data = train_test_split(*values_list,test_size=val_size,random_state=seed)
        data_train = splitted_data[::2]
        data_val = splitted_data[1::2]
    else:
        data_train, data_val = values_list,None
    train_dict = {}
    val_dict = {}
    for (key,_),value_train in zip(in_dict.items(),data_train):
        train_dict[key]=value_train
    if(val_size>0.0):
        for (key,_),value_val in zip(in_dict.items(),data_val):
            val_dict[key]=value_val
    if(return_type=="dataset"):
        train_dataset = Dataset.from_tensor_slices(train_dict).cache().shuffle(500).batch(batch_size,drop_remainder=True).prefetch(20)
        val_dataset = Dataset.from_tensor_slices(val_dict).cache().batch(batch_size,drop_remainder=True).prefetch(20)
        return train_dataset, val_dataset
    elif(return_type=="dict"):
        return train_dict, val_dict

def train_test_split_from_dict(in_dict, test_size, seed):
    keys = list(in_dict.keys())
    values = list(in_dict.values())
    new_values = train_test_split(*values, test_size=test_size, random_state=seed, shuffle=True)
    train_values = new_values[::2]
    test_values = new_values[1::2]
    train_dict = dict(zip(keys, train_values))
    test_dict = dict(zip(keys, test_values))
    return train_dict, test_dict