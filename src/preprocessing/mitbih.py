import numpy as np
from src.utils.preprocessing import *
from scipy import signal


# def preprocess_data_mit_bih(in_data,in_labels, dataset_params, **kwargs):
def preprocess_data_mit_bih(input_data, dataset_params, **kwargs):
    data = input_data["data"]
    labels  = input_data["labels"]
    fs = kwargs["fs"]
    batch_size = kwargs["batch_size"]
    split_ecg = kwargs["split_ecg"]
    classes = dataset_params["classes"]
    assert split_ecg==True
    data_preprocessed = copy.deepcopy(data.astype(np.float32))
    # labels_preprocessed = copy.deepcopy(in_labels)
    sos = signal.butter(5, [0.5,30], btype="bandpass", fs=fs, output='sos')
    data_preprocessed = signal.sosfilt(sos, data_preprocessed,axis=1).astype(np.float32)
    # if(fs!=360):
    #     signal.resample(data_preprocessed,)# TODO resample to fs if it's different from the original
    while(data_preprocessed.ndim<3): # Expand dimensions until we get 3 dimensions (sample, )
        data_preprocessed = np.expand_dims(data_preprocessed, axis=-1).astype(np.float32)
    # data_preprocessed = normalize_single_rows(data_preprocessed).astype(np.float32)
    # data_preprocessed = signal.sosfilt(sos, data_preprocessed,axis=1).astype(np.float32)
    if(split_ecg):
        sequence_length = kwargs["sequence_length"]
        # peaks_indexes = input_data["annotations"]["peaks"]
        peaks_indexes = []
        peaks_labels = []
        split_indexes = []
        for sample_ann_dict in input_data["annotations"]:
            peak_indexes_sample =sample_ann_dict["peaks"]
            middle_peaks_indexes = ((peak_indexes_sample[:-1] + peak_indexes_sample[1:]) / 2).astype(int)
            split_indexes.append(middle_peaks_indexes)
            peaks_labels.append(sample_ann_dict["labels"])
        # for ecg_peaks in peaks_indexes:
        #TODO: modify peak_indexes to cut between two peaks instead of on the peaks
        sections = split_ecg_in_waves(in_matrix = data_preprocessed,peaks_indexes=split_indexes)
        # Delete hearthbeats longer than sequence_length
        filtered_sections, filtered_labels = delete_sequences_longer_than(sections,peaks_labels,sequence_length)
        section_sizes = []
        section_sizes = [np.array([heartbeat.shape[0] for heartbeat in ecg_heartbeats]) for ecg_heartbeats in filtered_sections]
        # np.max([np.max(s_size) for s_size in section_sizes])
        
        # for ecg_heartbeats in samples_ecg_heartbeats:
        #     for heartbeat in ecg_heartbeats:
                
        #     section_sizes.append()
        # for i,ecg_heartbeats in enumerate(samples_ecg_heartbeats):
        #     ecg_heartbeats.pop(0)
        #     ecg_heartbeats.pop(-1)
        #     peaks_labels[i,0] =  #Should delete firt and last label if we pop the sections
        padded_section_list = pad_sequences(filtered_sections,sequence_length, center=False)
        # num_sections_per_sample = [sections.shape[0] for sections in padded_section_list]
        # padded_next_section_list = [np.roll(sections,shift=-1,axis=0) for sections in padded_section_list]
        # Delete to avoid morphological-only anomaly detection
        data_preprocessed = np.concatenate(padded_section_list,axis=0).astype(np.float32) 
        labels_preprocessed = np.concatenate(filtered_labels,axis=0).astype(int) 
        section_sizes= np.concatenate(section_sizes,axis=0).astype(np.int32) 
        
        # Remove -1 labels and correspondent heartbets and sizes
        wrong_labels_indexes = np.where(labels_preprocessed==-1)[0]
        # wrong_labels_indexes = labels_preprocessed>=0
        data_preprocessed = np.delete(arr=data_preprocessed,obj=wrong_labels_indexes,axis=0)
        labels_preprocessed = np.delete(arr=labels_preprocessed,obj=wrong_labels_indexes,axis=0)
        section_sizes = np.delete(arr=section_sizes,obj=wrong_labels_indexes,axis=0)
        
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



if __name__ == '__main__':
    pass