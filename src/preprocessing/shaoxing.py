import numpy as np
from src.utils.preprocessing import *
from src.preprocessing.ptb import preprocess_data_ptb_xl_fmm
def preprocess_data_shaoxing_fmm(input_data, dataset_params, **kwargs):
    # Simple preprocessing for shaoxing_fmm, which is loaded already processed
    # The preprocessing is the same as ptb_xl_fmm, so we reuse it
    return preprocess_data_ptb_xl_fmm(input_data,dataset_params,**kwargs)

if __name__ == '__main__':
    pass