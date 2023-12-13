import matplotlib.pyplot as plt
import re
import numpy as np

def add_subscript(key, subscript_symbol=""):
    # Add subscript to variables 
    parts = re.split(r'[_ ]', key)
    if len(parts) == 2 and parts[1].isdigit():
        return f"{parts[0]}{subscript_symbol}{parts[1]}" 

    return key

def replace_greek_symbols(key):
    # Replace Greek letter names with symbols
    replacements = {
        'alpha': 'α',
        'beta': 'β',
        'omega': 'ω'
    }
    for symbol, name in replacements.items():
        if symbol in key:
            key = key.replace(symbol, name)
    return key

def plot_scalar_dictionary(data, std=None, ylabel="Parameters", xlabel="Correlation", title="Correlation between original and predicted FMM parameters",fontsize=12):
    # Plot dictionary stored in data as bar plot
    # Dictionary should have string keys and scalar numerical (e.g., float) items
    plt.figure()
    names = list(data.keys())
    values = list(data.values())

    # Apply the functions sequentially to modify the keys
    names = [replace_greek_symbols(add_subscript(name)) for name in names]

    plt.barh(names, values)  # Swapped x and y axes here
    if(std is not None):
        std_names = list(std.keys())
        std_values = list(std.values())
        std_names = [replace_greek_symbols(add_subscript(name)) for name in std_names]
        assert std_names==names
        plt.errorbar(values, std_names, xerr=std_values, fmt="o", color="r", markersize=2)  
    plt.ylabel(ylabel, size=fontsize)  # Swapped x and y axis labels
    plt.xlabel(xlabel, size=fontsize)  # Swapped x and y axis labels
    plt.title(title)
    plt.yticks(rotation=0)  # Swapped x and y axis ticks rotation
    plt.gca().invert_yaxis() 
    # plt.show()

def plot_curve(x, y, z, elevation_angle=30, azimuthal_angle=30):
    """
    Plots a curve using the given sequences of integers for x, y, and z coordinates.
    
    Args:
        x (list): Sequence of integers representing x coordinates.
        y (list): Sequence of integers representing y coordinates.
        z (list): Sequence of integers representing z coordinates.
    """
    # Check if all input sequences have the same length
    if len(x) != len(y) or len(y) != len(z):
        print("Error: Input sequences must have the same length.")
        return
    # Plot the curve using x, y, and z coordinates
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    # Set labels for x, y, and z axes
    ax.set_xlabel('Lead II')
    ax.set_ylabel('Lead II\'')
    ax.set_zlabel('Lead V2')
    ax.view_init(elev=elevation_angle, azim=azimuthal_angle)

def plot_roc(fpr:np.ndarray, tpr:np.ndarray, roc_auc:float, filename:str=None):
    """Plot roc curve from fpr and tpr

    Args:
        fpr (np.ndarray): False positive rate array
        tpr (np.ndarray): True positive rate array
        roc_auc (float): Area under the roc
        filename (str): Save file name. Defaults to None.
    """
    lw = 2
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.4f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    if(filename):
        plt.savefig(filename+".png")
        plt.savefig(filename+".eps")

def plot_group_bars(group_names, data_lists, bar_names, colors=None, 
                    xlabel='', ylabel='', title='',
                    axis_fontsize=12, legend_fontsize=10, tick_fontsize=10):
    num_groups = len(group_names)
    num_bars = len(data_lists)
    bar_width = 0.8 / num_bars  # Adjust the width dynamically
    index = np.arange(num_groups)

    fig, ax = plt.subplots()
    if(colors is not None):
        for i in range(num_bars):
            ax.bar(index + i * bar_width, data_lists[i], bar_width, label=bar_names[i], color=colors[i])
    else:
        for i in range(num_bars):
            ax.bar(index + i * bar_width, data_lists[i], bar_width, label=bar_names[i])
    ax.set_xlabel(xlabel, fontsize=axis_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_fontsize)
    ax.set_title(title, fontsize=axis_fontsize)
    ax.set_title(title)
    ax.set_xticks(index + bar_width * (num_bars - 1) / 2)
    ax.set_xticklabels(group_names, fontsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
