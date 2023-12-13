import sklearn
import numpy as np
from tensorflow.data import Dataset
import matplotlib.pyplot as plt 
from src.plot.vaeplot import plot_clusters
from src.plot.general import plot_roc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List
import tensorflow as tf

def get_loss_score(in_vae, data_dict, batch_size, classes, num_samples=-1):
    # Compute losses for all the classes
    predict_results_dict = in_vae.predict(Dataset.from_tensor_slices(data_dict).batch(batch_size,drop_remainder=True).take(num_samples)) #train_dict, batch_size=batch_size
    original = predict_results_dict["data"]
    reconstruction = predict_results_dict["predicted_data"]
    # Compute and display losses for all the classes
    labels = data_dict["labels"][:original.shape[0]]
    # if(isinstance(original,List)):
    #     original = original[0]
    #     reconstruction = reconstruction[0]
    #     labels = np.repeat(labels,repeats=7)
    #     labels = labels[:reconstruction.shape[0]]
        
    reconstruction_loss =  masked_mse(original, reconstruction) #np.sum(np.square(reconstruction-original),axis=1)
    class_loss_dict = {}
    class_mean_loss = np.zeros_like(classes, dtype=np.float32)
    class_std_loss = np.zeros_like(classes, dtype=np.float32)
    for i,class_name in enumerate(classes):
        # plt.figure()
        reconstruction_loss_class = reconstruction_loss[labels==i]
        # print(reconstruction_loss_class.shape)
        # plt.hist(reconstruction_loss,label=class_name,bins=2000,alpha=0.4, density=True)
        # class_loss_dict[f"{class_name} mean"] = np.mean(reconstruction_loss_class,axis=0)
        class_mean_loss[i] = np.mean(reconstruction_loss_class,axis=0)
        # print(class_loss_dict[f"{class_name} mean"])
        # class_loss_dict[f"{class_name} std"] = np.std(reconstruction_loss_class,axis=0)
        class_std_loss[i] = np.std(reconstruction_loss_class,axis=0)
    class_loss_dict = {"mean":class_mean_loss.tolist(), "std":class_std_loss.tolist()}
    return class_loss_dict


def get_confusion_matrix(in_vae, data_dict, batch_size, threshold, classes, normal_class, filename=None):
    predict_results_dict = in_vae.predict(Dataset.from_tensor_slices(data_dict).batch(batch_size,drop_remainder=True))
    original = predict_results_dict["data"]
    reconstruction = predict_results_dict["predicted_data"]
    labels = data_dict["labels"][:original.shape[0]]
    reconstruction_loss = masked_mse(original, reconstruction)
    #Compute normal and abnormal indexes
    real_abnormal_indexes = labels!=normal_class
    # normal_indexes = labels==normal_class
    # Compute which samples have loss greater than threshold
    detected_abnormal_indexes = reconstruction_loss>threshold
    # Create confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true = real_abnormal_indexes,
                                                        y_pred=detected_abnormal_indexes)
    return confusion_matrix

def get_roc_auroc(in_vae, data_dict, batch_size, normal_class, filename=None):
    predict_results_dict = in_vae.predict(Dataset.from_tensor_slices(data_dict).batch(batch_size, drop_remainder=True))
    original = predict_results_dict["data"]
    reconstruction = predict_results_dict["predicted_data"]
    labels = data_dict["labels"]
    # if(isinstance(original,List)):
    #     original = original[0]
    #     reconstruction = reconstruction[0]
    #     labels = np.repeat(labels,repeats=7)
    #     labels = labels[:reconstruction.shape[0]]    
    # reconstruction_loss = np.sum(np.square(reconstruction-original),axis=1)
    reconstruction_loss = masked_mse(original, reconstruction)
    # positive_classes = list(range(len(classes))).remove(normal_class)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels!=normal_class, y_score=reconstruction_loss)
    roc_auc = sklearn.metrics.auc(fpr,tpr)
    plot_roc(fpr,tpr,roc_auc,filename)
    roc_dict = { "fpr":fpr.tolist(), "tpr":tpr.tolist(), "thresholds":thresholds.tolist(),"roc_auc":roc_auc}
    return roc_dict

def plot_pca(in_vae, data_dict, batch_size, classes, filename=None):
    predict_results_dict = in_vae.predict(Dataset.from_tensor_slices(data_dict).batch(batch_size,drop_remainder=True)) #train_dict, batch_size=batch_size
    z_mean = predict_results_dict["z_mean"]
    labels = data_dict["labels"]
    # if(isinstance(z_mean,List)):
    #     z_mean = np.concatenate(z_mean,axis=1)
    #     labels = np.repeat(labels,repeats=7)
    #     labels = labels[:z_mean.shape[0]]  
    # Plot clusters data (latent space, only mean variables)
    pca = PCA(n_components=2)
    pca.fit(z_mean)
    transformed_z_mean = pca.transform(z_mean)
    plot_clusters(transformed_z_mean,labels,leg=classes) #[:11648]
    plt.title("PCA-reduced mean parameter of latent distribution")
    plt.xlabel("PCA dimension 1")
    plt.ylabel("PCA dimension 2")
    if(filename):
        # filename = get_workspace_path("clusters_plot")
        plt.savefig(filename+".png")
        plt.savefig(filename+".eps")
        
def plot_tsne(in_vae, data_dict, batch_size, classes, filename=None):
    predict_results_dict = in_vae.predict(Dataset.from_tensor_slices(data_dict).batch(batch_size,drop_remainder=True)) #train_dict, batch_size=batch_size
    z_mean = predict_results_dict["z_mean"]
    labels = data_dict["labels"]
    # if(isinstance(z_mean,List)):
    #     z_mean = np.concatenate(z_mean,axis=1)
    #     labels = np.repeat(labels,repeats=7)
    #     labels = labels[:z_mean.shape[0]]   
    # Plot clusters data (latent space, only mean variables)
    pca_before_tsne = PCA(n_components=10)
    pca_before_tsne.fit(z_mean)
    pca_transformed_predicted_val = pca_before_tsne.transform(z_mean)
    tsne = TSNE()
    tsne_transformed_predicted_val = tsne.fit_transform(pca_transformed_predicted_val)
    plot_clusters(tsne_transformed_predicted_val,labels,leg=classes) 
    if(filename):
        # filename = get_workspace_path("clusters_plot")
        plt.savefig(filename+".png")
        plt.savefig(filename+".eps")

def plot_one_ecg_for_class(in_vae=None, data_dict=None, batch_size:int=None, classes=None, fs=None, filename=None,lead=0):
    if(in_vae):
        predict_results_dict = in_vae.predict(Dataset.from_tensor_slices(data_dict).batch(batch_size,drop_remainder=True)) #train_dict, batch_size=batch_size
        original = predict_results_dict["data"]
        reconstruction = predict_results_dict["predicted_data"]
    else:
        original = data_dict["inputs"]
    samples_to_plot_list = []
    if(data_dict["labels"] is not None):
        for i in range(len(classes)):
            class_indexes = np.where(data_dict["labels"]==i)[0]
            if(len(class_indexes)>0):
                num_class_examples = np.shape(class_indexes)[0]
                random_index = np.random.randint(low=0, high=num_class_examples)
                index_to_plot = class_indexes[random_index]
            else:
                index_to_plot = None # Branch taken if class is empty->add none to samples_to_plot_list (no plot)
            samples_to_plot_list.append(index_to_plot)
        for sample_index in samples_to_plot_list:
            if(sample_index is not None):
                plt.figure()
                class_label = classes[data_dict["labels"][sample_index,].astype(int)]
                sequence_len = np.squeeze(original[sample_index,:,lead]).shape[0]
                xaxis = np.arange(1,sequence_len+1)/fs
                plt.plot(xaxis, np.squeeze(original[sample_index,:,lead]), label="Original ECG")
                if(in_vae):
                    plt.plot(xaxis,np.squeeze(reconstruction[sample_index,:,lead]), label="Predicted ECG")
                    plt.title(f"Model prediction vs original data sample: {sample_index}, class: {class_label}, lead: {lead}")
                else:
                    plt.title(f"Data sample index: {sample_index}, class: {class_label}, lead: {lead}")
                plt.legend()
                plt.xlabel("Time [s]")
                plt.ylabel("ECG")
                if(filename):
                    filename2 = f"{filename}_{sample_index}_lead_{lead}"
                    plt.savefig(filename2+".png")
                    plt.savefig(filename2+".eps")


def masked_mse(data,reconstruction):
    # Return mse loss vector where the data is different from 0
    mask = data!=0
    mask = tf.cast(mask, dtype=data.dtype)
    squared_difference = tf.math.squared_difference(reconstruction, data)
    # reconstruction_loss = tf.reduce_mean(tf.reduce_sum(squared_difference,axis=[1,2]),axis=0)
    squared_difference *= mask
    # Do the division by sum of mask values to make the mean over only the non-padding values 
    # reconstruction_loss = tf.reduce_sum(tf.reduce_sum(squared_difference,axis=[1,2]),axis=0)/tf.reduce_sum(mask)
    # reconstruction_loss = tf.reduce_sum(squared_difference,axis=[1,2])/tf.reduce_sum(mask,axis=(1,2))
    max_axis = tf.size(tf.shape(data))
    # for i in range(max_axis)
    x = tf.reduce_sum(squared_difference,axis=max_axis-1)
    x = tf.math.divide_no_nan(x,tf.reduce_sum(mask,axis=max_axis-1))
    # if(tf.cond(tf.math.equal(max_axis,3),true_fn=lambda:1,false_fn=lambda:0)):
    # if(max_axis==3):
    #     reconstruction_loss = tf.reduce_sum(x,axis=1)
    # elif(max_axis==2):#tf.cond(tf.math.equal(max_axis,2),true_fn=lambda:1,false_fn=lambda:0)
    #     reconstruction_loss = x
    reconstruction_loss = tf.reduce_sum(x,axis=1)
    # else:
    #     raise ValueError("data number of dimensions should be 2 or 3")
    # if(tf.size(tf.shape(data))==3):
    #     x = tf.reduce_sum(data,axis=2)
    #     x = x/tf.reduce_sum(mask,axis=2)
    #     reconstruction_loss = tf.reduce_sum(x,axis=1)
    # else:
    #     raise NotImplementedError()
    return reconstruction_loss

def mse_timeseries(data,reconstruction):
    squared_difference = tf.math.squared_difference(data, reconstruction) # SE between inputs and reconstructions
    max_axis = tf.size(tf.shape(data))
    x = tf.reduce_mean(squared_difference,axis=max_axis-1) # Average all the MSE between all the features/leads
    reconstruction_loss = tf.reduce_sum(x,axis=1) # Average the values per timestep
    return reconstruction_loss

def weighted_mean_squared_error(x, y, w):
    squared_diff = tf.square(x - y)
    # mse = tf.linalg.matmul(squared_diff,tf.expand_dims(w,axis=-1))/tf.shape(squared_diff,out_type=float)[-1]
    weighted_squared_diff = squared_diff * w
    mse = tf.reduce_mean(weighted_squared_diff, axis=-1)
    return {"error":mse,"error_vector":squared_diff,"weighted_error_vector":weighted_squared_diff}

def circular_squared_error(x,y):
    return (tf.square(tf.cos(x)-tf.cos(y)) + tf.square(tf.sin(x)-tf.sin(y)))

def squared_error(x,y):
    return tf.square(x-y)

def circular_weighted_mean_square_error(x:tf.Tensor, y:tf.Tensor, w:tf.Tensor, c:tf.Tensor)->tf.Tensor:
    """Compute MSE between x and y, where each pair in tensors x,y is weighted according to w. 
    It also supports circular variables.

    Args:
        x (tf.Tensor): first input tensor
        y (tf.Tensor): second input tensor
        w (tf.Tensor): weight, same dimension of the last dimension of x,y
        c (tf.Tensor): float array where each element specify if the pair (x_i,y_i) is circular (c_i=1) or not (c_i=0)

    Returns:
        tf.Tensor: mse loss
    """
    se = squared_error(x,y)
    cse = circular_squared_error(x,y)
    e = c*cse + (1-c)*se
    weighted_error = e * w
    mse = tf.reduce_mean(weighted_error, axis=-1)
    return {"error":mse,"error_vector":e,"weighted_error_vector":weighted_error}

if __name__=='__main__':
    pass