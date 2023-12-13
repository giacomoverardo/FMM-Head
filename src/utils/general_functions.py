import pandas as pd
import json
import numpy as np
import shutil
import os 
from typing import List, Tuple
import urllib
import wget
import random
import tensorflow as tf
import pickle

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def delete_folder(path):
    '''
    Delete results folder if present
    '''
    if(os.path.exists(path)):
        shutil.rmtree(path)

def get_directories(path:str)->List[str]:     
    dirList = [] 
    for element in os.listdir(path):
        if (os.path.isdir(os.path.join(path,element))):
            dirList.append(element)
    return dirList

def create_folder(path:str, delete_previous:bool=False):
    '''
    Create a folder if not already present
    '''
    if(delete_previous):
        shutil.rmtree(path)
    if(not os.path.exists(path)):
        os.makedirs(path)
        # os.mkdir(path)

def scalar_to_list(x, listSize):
    if(isinstance(x, (list,np.ndarray))):
        if(len(x)==listSize):
            return x
        else:
            if(len(x)==1):
                return [x[0] for _ in range(listSize)]
            else:
                raise ValueError("If input x is a List, it should have size listSize. Input values: {0}, {1}".format(x,listSize))
    else:
        return [x for _ in range(listSize)]

def scalar_to_np_array(x,array_size):
    out_list = scalar_to_list(x,array_size)
    return np.array(out_list)

def save_history_to_json(historyDict,filename:str):
    hist_df = pd.DataFrame(historyDict) 
    with open(filename, mode='w') as f:
        hist_df.to_json(f)

def save_dict_to_binary(dictionary, filename):
    """
    Save a dictionary to a pickle file.
    
    Args:
        dictionary (dict): The dictionary to be saved.
        filename (str): The name of the file to save the dictionary to.
    """
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

def load_dict_from_binary(filename):
    """
    Load a dictionary from a pickle file.
    
    Args:
        filename (str): The name of the file to load the dictionary from.
        
    Returns:
        dict: The loaded dictionary.
    """
    with open(filename, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict


def get_download_file_name(url):
    # Get file name of the file name that will be downloaded with wget
    _,headers =urllib.request.urlretrieve(url)
    return wget.filename_from_headers(headers)

def set_ml_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # TF 2.1
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_all_subfolders(folder_path):
    subfolder_paths = []

    for root, dirs, _ in os.walk(folder_path):
        for dir_name in dirs:
            subfolder_paths.append(os.path.join(root, dir_name))

    return subfolder_paths

# def get_subfolders_at_depth(folder_path, depth):
#     subfolder_paths = []

#     for root, dirs, _ in os.walk(folder_path):
#         current_depth = root[len(folder_path) + len(os.path.sep):].count(os.path.sep)
#         if current_depth == depth and current_depth > 0:  # Exclude the root path itself
#             subfolder_paths.extend([os.path.join(root, dir_name) for dir_name in dirs])

#     return subfolder_paths


def get_subfolders_at_depth(folder_path, depth):
    #TODO: DOES NOT WORK FOR DEPTH DIFFERNT THAN 1
    assert depth==1
    # Source: https://stackoverflow.com/questions/7159607/list-directories-with-a-specified-depth-in-python
    path = folder_path
    path = os.path.normpath(path)
    res = []
    for root,dirs,files in os.walk(path, topdown=True):
        cuurent_depth = root[len(path) + len(os.path.sep)-1:].count(os.path.sep)
        if cuurent_depth == depth:
            # We're currently two directories in, so all subdirs have depth 3
            res += [os.path.join(root, d) for d in dirs]
            # dirs[:] = [] # Don't recurse any deeper
    # res.pop(0)
    return res

def format_number_with_sign(number, decimals=0):
    if number >= 0:
        return f'+{number:.{decimals}f}'
    else:
        return f'{number:.{decimals}f}'
    
def format_numbers_with_sign(numbers):
    formatted_numbers = []
    for num in numbers:
        formatted_numbers.append(format_number_with_sign(num))
    return formatted_numbers

if __name__ == '__main__':
    pass