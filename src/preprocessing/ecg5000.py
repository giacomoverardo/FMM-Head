import numpy as np
from src.utils.preprocessing import *

# def preprocess_data_ecg5000(in_data,in_labels, dataset_params, **kwargs):
def preprocess_data_ecg5000(input_data, dataset_params, **kwargs):
    data = input_data["data"]
    labels  = input_data["labels"]
    sizes = input_data["sizes"]
    fs = kwargs["fs"]
    batch_size = kwargs["batch_size"]
    classes = dataset_params["classes"]
    # rythm = params["rythm"]
    data_preprocessed = copy.deepcopy(data.astype(np.float32))
    labels_preprocessed = copy.deepcopy(labels)
    # sos = scipy.signal.butter(5, [0.5,30], btype="bandpass", fs=fs, output='sos')
    # data_preprocessed = scipy.signal.sosfilt(sos, data_preprocessed,axis=1).astype(np.float32)
    data_preprocessed = normalize_single_rows(data_preprocessed).astype(np.float32)
    data_preprocessed = data_preprocessed - np.expand_dims(np.mean(data_preprocessed,axis=1),axis=-1)
    data_preprocessed = np.roll(data_preprocessed,shift=70,axis=1)
    while(data_preprocessed.ndim<3): # Expand dimensions until we get 3 dimensions (sample, )
        data_preprocessed = np.expand_dims(data_preprocessed, axis=-1).astype(np.float32)
        # correct_indices = correct_indices[correct_indices<data_preprocessed.shape[0]]
    data_preprocessed,labels_preprocessed, sizes_preprocessed = \
        reduce_to_batch([data_preprocessed,labels_preprocessed, sizes],in_batch_size=batch_size)
    num_samples_per_class = [np.sum(labels_preprocessed==i) for i in range(len(classes))]
    print(f"Number of samples per class: {num_samples_per_class}")
    data_dict = {"inputs":data_preprocessed,"labels":labels_preprocessed, "sizes":sizes_preprocessed}       
    # data_dataset = Dataset.from_tensor_slices(data_dict).shuffle(500).batch(batch_size,drop_remainder=True) 
    return data_dict

if __name__ == '__main__':
    pass