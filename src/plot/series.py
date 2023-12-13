import matplotlib.pyplot as plt
import numpy as np

def plot_series_list(inSeriesList,legend=None,savepath=None): 
    plt.figure()
    for inSeries in inSeriesList:
        plt.plot(inSeries)
    plt.legend(legend)
    if(savepath):
        plt.savefig(savepath)
    else:
        plt.show()
        
def subplot_ecg(ecg_data:np.ndarray, labels:np.ndarray=None, num_to_plot:int=9, lead:int=0,indexes=None):
    plot_per_row = int(np.sqrt(num_to_plot))
    num_to_plot = np.square(plot_per_row,dtype=int)
    if(indexes is None):
        indexes_to_plot = np.random.randint(low=0, high=np.shape(ecg_data)[0],size=(num_to_plot,))
    else:
        indexes_to_plot = indexes
    add_class_in_title = labels is not None
    if(add_class_in_title):
        assert ecg_data.shape[0] == labels.shape[0]
    count = 1
    for i_plot in indexes_to_plot:
        # x,y = np.unravel_index(count,(plot_per_row,plot_per_row))
        ax1 = plt.subplot(plot_per_row, plot_per_row, count)
        count += 1
        if(ecg_data.ndim==3):
            to_plot = ecg_data[i_plot,:,lead]
        elif(ecg_data.ndim==2):
            to_plot = ecg_data[i_plot,:]
        ax1.plot(to_plot)
        ax1.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        title = f"ECG {i_plot}"
        title = f"{title}, class: {int(labels[i_plot])}" if add_class_in_title else title
        ax1.set_title(title,fontdict={'fontsize': 6})