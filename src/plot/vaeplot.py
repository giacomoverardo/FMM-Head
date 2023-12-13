import matplotlib.pyplot as plt
import numpy as np

def plot_latent_space(vae, num_samples, n=30, figsize=15, pca=None):
    # display a n*n 2D manifold of digits
    # digit_size = 28
    # num_samples = 1250
    scale = 1.0
    figure = np.zeros((num_samples * n, n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    figure, axis =plt.subplots(len(grid_x), len(grid_y))
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            if(pca):
                z_sample = pca.inverse_transform([xi, yi]) # NOT USED: PCA IS ON MEAN PARAMETERS, NOT SAMPLES!!!
                z_sample = [z_sample]
            else:
                z_sample = np.array([[xi, yi]])  
            x_decoded = vae.decoder.predict(z_sample)
            time_series = x_decoded[0].reshape(num_samples, 1)
            axis[i,j].plot(time_series)
            axis[i,j].set_title(f"({np.round(xi,2)},{np.round(yi,2)})",fontsize=6,y=0.9)
            axis[i,j].axes.xaxis.set_ticklabels([])
            axis[i,j].axes.yaxis.set_ticklabels([])
    return figure
            # figure[
            #     i * num_samples : (i + 1) * num_samples,
            #     j * 1 : (j + 1) * 1,
            # ] = time_series

    # plt.figure(figsize=(figsize, figsize))
    # start_range = num_samples // 2
    # end_range = n * num_samples + start_range
    # pixel_range = np.arange(start_range, end_range, num_samples)
    # sample_range_x = np.round(grid_x, 1)
    # sample_range_y = np.round(grid_y, 1)
    # plt.xticks(pixel_range, sample_range_x)
    # plt.yticks(pixel_range, sample_range_y)
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    # plt.imshow(figure, cmap="Greys_r")
    # plt.show()

def plot_label_clusters(vae, data, labels,xlim=None,ylim=None):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    if(xlim):
        plt.xlim(xlim)
    if(ylim):
        plt.ylim(ylim)
    plt.show()

def plot_clusters(data,labels,xlim=None,ylim=None,leg=None):
    # Data should only have two features to be plot
    assert len(data.shape)==2
    assert data.shape[1]==2
    unique_labels = np.unique(labels) # Returns classes in order
    clusters_data = [data[labels==current_label,:] for current_label in unique_labels] 
    clusters_data = np.array(clusters_data)
    plt.figure(figsize=(12, 10))
    # plt.scatter(data[:, 0], data[:, 1], c=labels)
    for i in range(len(unique_labels)):
        plt.scatter(clusters_data[i][:,0], clusters_data[i][:,1]) #,label=leg[i]
    plt.legend(leg)
    # plt.legend(['Normal','HB','PMI', 'MI','COVID'])
    # plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    if(xlim):
        plt.xlim(xlim)
    if(ylim):
        plt.ylim(ylim)