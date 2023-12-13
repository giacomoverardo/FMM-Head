import numpy as np
from src.utils.preprocessing import *
from scipy import signal
from src.utils.fmm import get_A_indexes

# def preprocess_data_ptb(in_data,in_labels, dataset_params, **kwargs):
def preprocess_data_ptb(input_data, dataset_params, **kwargs):
    data = input_data["data"]
    labels  = input_data["labels"]
    fs = kwargs["fs"]
    batch_size = kwargs["batch_size"]
    split_ecg = kwargs["split_ecg"]
    classes = dataset_params["classes"]
    # rythm = params["rythm"]
    data_preprocessed = copy.deepcopy(data.astype(np.float32))
    labels_preprocessed = copy.deepcopy(labels)
    sos = signal.butter(5, [0.5,30], btype="bandpass", fs=fs, output='sos')
    data_preprocessed = signal.sosfilt(sos, data_preprocessed,axis=1).astype(np.float32)
    while(data_preprocessed.ndim<3): # Expand dimensions until we get 3 dimensions (sample, )
        data_preprocessed = np.expand_dims(data_preprocessed, axis=-1).astype(np.float32)
    data_preprocessed = normalize_single_rows(data_preprocessed).astype(np.float32)
    if(split_ecg):
        sequence_length = kwargs["sequence_length"]#int(1.6*fs) 
        peaks_indexes = get_peaks_indexes_wavelet(data_preprocessed[:,:,0],minimum_height=0.01, peak_to_peak_distance=int(0.4*fs))
        # for ecg_peaks in peaks_indexes:
        #TODO: modify peak_indexes to cut between two peaks instead of on the peaks
        sections_list = split_ecg_in_waves(in_matrix = data_preprocessed,peaks_indexes=peaks_indexes)
        # Delete samples which have sections with too small or too high sizes
        min_num_sections = 5
        filtered_section_list, filtered_labels, correct_indices, seq_sizes = remove_incorrectly_divided_samples(sections_list,labels,min_ecg_time=None, #int(0.7*fs)
                                                                                    max_ecg_time=sequence_length, min_num_sections=2+min_num_sections)
        print("Supposed correctly segmented ecg: {}".format(correct_indices.shape[0]))
        # Delete first and last section for each sample
        for sections in filtered_section_list:
            sections.pop(0)
            sections.pop(-1)
        padded_section_list = pad_sequences(filtered_section_list,sequence_length, center=False)
        num_sections_per_sample = [sections.shape[0] for sections in padded_section_list]
        # padded_next_section_list = [np.roll(sections,shift=-1,axis=0) for sections in padded_section_list]
        # Delete to avoid morphological-only anomaly detection
        data_preprocessed = np.concatenate(padded_section_list,axis=0).astype(np.float32) 
        labels_preprocessed = np.repeat(filtered_labels,repeats=num_sections_per_sample)
        section_sizes= np.concatenate(seq_sizes,axis=0).astype(np.int32) 
        
        while(data_preprocessed.ndim<3): # Expand dimensions until we get 3 dimensions (sample, )
            data_preprocessed = np.expand_dims(data_preprocessed, axis=-1).astype(np.float32)
            # correct_indices = correct_indices[correct_indices<data_preprocessed.shape[0]]
        data_preprocessed,labels_preprocessed,section_sizes = \
            reduce_to_batch([data_preprocessed,labels_preprocessed,section_sizes],in_batch_size=batch_size)
        num_samples_per_class = [np.sum(labels_preprocessed==i) for i in range(len(classes))]
        print(f"Number of samples per class: {num_samples_per_class}")
        data_dict = {"inputs":data_preprocessed,"labels":labels_preprocessed,"sizes":section_sizes}       
        # data_dataset = Dataset.from_tensor_slices(data_dict).shuffle(500).batch(batch_size,drop_remainder=True) 
        return data_dict
    while(data_preprocessed.ndim<3): # Expand dimensions until we get 3 dimensions (sample, )
        data_preprocessed = np.expand_dims(data_preprocessed, axis=-1).astype(np.float32)
    # correct_indices = correct_indices[correct_indices<data_preprocessed.shape[0]]
    data_preprocessed,labels_preprocessed = \
        reduce_to_batch([data_preprocessed,labels_preprocessed],in_batch_size=batch_size)
    num_samples_per_class = [np.sum(labels_preprocessed==i) for i in range(len(classes))]
    print(f"Number of samples per class: {num_samples_per_class}")
    data_dict = {"inputs":data_preprocessed,"labels":labels_preprocessed}       
    # data_dataset = Dataset.from_tensor_slices(data_dict).shuffle(500).batch(batch_size,drop_remainder=True) 
    return data_dict

def preprocess_data_ptb_xl_fmm(input_data, dataset_params, **kwargs):
    #Simple preprocessing for ptb_xl_fmm, which is loaded already processed
    data = input_data["data"]
    labels  = input_data["labels"]
    sizes  = input_data["sizes"]
    coefficients  = input_data["coefficients"]
    coefficients_ang  = input_data["coefficients_ang"]
    fs = kwargs["fs"]
    batch_size = kwargs["batch_size"]
    split_ecg = kwargs["split_ecg"]
    classes = dataset_params["classes"]
    data_preprocessed = copy.deepcopy(data.astype(np.float32))
    labels_preprocessed = copy.deepcopy(labels)
    while(data_preprocessed.ndim<3): # Expand dimensions until we get 3 dimensions (sample, )
        data_preprocessed = np.expand_dims(data_preprocessed, axis=-1).astype(np.float32)
    data_preprocessed,labels_preprocessed,sizes,coefficients,coefficients_ang = \
        reduce_to_batch([data_preprocessed,labels_preprocessed,sizes,coefficients,coefficients_ang],in_batch_size=batch_size)
    num_samples_per_class = [np.sum(labels_preprocessed==i) for i in range(len(classes))]
    print(f"Number of samples per class: {num_samples_per_class}")
    data_dict = {"inputs":data_preprocessed,"labels":labels_preprocessed, "sizes":sizes,"coefficients":coefficients,"coefficients_ang":coefficients_ang}       
    return data_dict


if __name__ == '__main__':
    pass