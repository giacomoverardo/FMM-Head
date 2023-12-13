# %%
import logging 
import hydra
import numpy as np
from src.utils.metrics import *
from src.plot.series import *
from src.plot.vaeplot import *
from src.utils.math import *
from src.utils.general_functions import *
from src.utils.preprocessing import *
from src.utils.nn import *
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from hydra.utils import get_original_cwd, instantiate, call
from omegaconf import OmegaConf
import pickle
import os
from rpy2.robjects.packages import STAP
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.rinterface_lib.embedded import RRuntimeError
from multiprocessing import Pool, cpu_count
import tqdm
from src.utils.fmm import format_FMM_wave_coefficients
import signal
import time
logger = logging.getLogger(__name__)
os.environ["HYDRA_FULL_ERROR"] = "1"


dataset_split = "train" # Subfolder of dataset (train or test for ptb_xl, all for shaoxing)
dataset_name = "ptb_xl_fmm" # ptb_xl_fmm or shaoxing_fmm
freq_ds = 100 # Dataset sampling frequency (100 for ptb_xl, 500 for shaoxing)
# Add your datapath here, where the unpreprocessed dataset is located (e.g. "./data")
data_path =  ??? # Add the path of your datasets here (same as configuration file)
r_package_path = "???/FMMECG3D" # Add the path of the FMMECG3D code location here (same as configuration file)
save_files=True # save file or not (False to only compute extraction time)
prep_dataset_path = os.path.join(data_path, dataset_name, dataset_split) # The processed dataset with FMM coefficients will be saved here
do_fmm_fit = False
# %%
prep_package_name = os.path.join(r_package_path,"requiredFunctionsPreprocessing_v4.1.R")
with open(prep_package_name, 'r',encoding='latin-1') as f:
    r_code = f.read()
prep_package = STAP(r_code, "prep_pkg")
fmm_package_name = os.path.join(r_package_path,"auxMultiFMM_ECG.R")
with open(fmm_package_name, 'r',encoding='latin-1') as f:
    r_code = f.read()
fmm_package = STAP(r_code, "fmm_pkg")

def save_fmm_parameters(beat):
    selected_beat_data = beat["data"]
    selected_beat_peak_index = beat["peak_index"]
    sample_id = beat["sample_index"]
    beat_id = beat["beat_index"]
    ecg_id = beat["ecg_id"]
    patient_id = beat["patient_id"]
    if(do_fmm_fit):
        # Save beat to file with fmm coefficients
        starting_time = time.time()
        fmm_parameters= fmm_package.fitMultiFMM_ECG(vDataMatrix = selected_beat_data,
                                            annotation = selected_beat_peak_index,parallelize=False)
        ending_time = time.time()
        elapsed_time = ending_time - starting_time
        #Check that all leads have the correct wave names
        wave_name_list = ["R","T","P","S","Q"]
        for fmm_lead_parameters in fmm_parameters:
            error = False
            try:
                for wave_name in wave_name_list:
                    assert wave_name in fmm_lead_parameters.rownames
            except AssertionError as e:
                error=True
                break
        if(not(error)):
            if(save_files):
                parameters_dict = format_FMM_wave_coefficients(fmm_parameters)
                return_dict = {"coefficients":parameters_dict,**beat}
                with open(os.path.join(prep_dataset_path,f"sample_{ecg_id}_beat_{beat_id}"), 'wb') as f:
                    pickle.dump(return_dict,f)
            return elapsed_time
        elif(error):
            return None
    else:
        # Save only beat without fitting fmm to file
        with open(os.path.join(prep_dataset_path,f"sample_{ecg_id}_beat_{beat_id}"), 'wb') as f:
            pickle.dump(beat,f)    

def timeout_handler(signum, frame):
    raise TimeoutError("Function timed out!")
# %%
def preprocess_data_multi_core(r_input):
        r_data = r_input["data"]
        r_label = r_input["label"]
        sample_info = r_input["info"]
        error = 1
        data_preprocessed = None
        metadata_preprocessed = None
        # Set the signal handler for SIGALRM (alarm signal)
        signal.signal(signal.SIGALRM, timeout_handler)
        timeout_seconds = 30
        try:
            # Set an alarm to trigger the alarm signal after the specified time
            signal.alarm(timeout_seconds)
            r_preprocessed_data = prep_package.givePreprocessing_git(dataIn=r_data,freqHz=freq_ds)
            signal.alarm(0)  # Cancel the alarm if the function completes successfully
            data_preprocessed = r_preprocessed_data[1] # Save preprocessed data 
            metadata_preprocessed = r_preprocessed_data[0] # Save metadata such as beat start/end, R peack etc. Use list since number of  peaks is variable
            error = 0
            if(not(isinstance(metadata_preprocessed,np.ndarray))): # In some cases the R code produce wrong metadata
                error = 1
        except RRuntimeError as e:
            pass
        except TimeoutError:
            print("Timeout occurred!")
        except:
            pass
        return {"data": data_preprocessed, "label": r_label, "metadata":metadata_preprocessed, "error":error, "sample_info": sample_info}
def extract_beat_list(mp_results):
    results_list = []
    for result_dict in mp_results:
        if result_dict["error"]==0:
            results_list.append((result_dict["data"],result_dict["label"],result_dict["metadata"], result_dict["sample_info"]))
        else:
            pass
    beat_list = []
    for sample_index,(d_prep,l_prep,meta_prep,sample_info) in enumerate(results_list):
        for meta in meta_prep[1:-1]: # Extract all beats from samples except first and last
            beat_index = int(meta[0])
            anno_ref = int(meta[1])
            beat_start = int(meta[2])
            beat_end = int(meta[3])
            beat_wave = d_prep[beat_start:beat_end,:]
            beat_list.append({"data":beat_wave, 
                            "label":l_prep,
                            "peak_index":anno_ref-beat_start+1,
                            "len":beat_end-beat_start+1,
                            "sample_index":sample_index,
                            "beat_index":beat_index,
                            **sample_info
                            })
            
    return beat_list
@hydra.main(version_base="1.3", config_path="conf", config_name="generate_fmm_ds") 
def my_app(cfg) -> None:
    numpy2ri.activate()
    pandas2ri.activate()
    set_ml_seeds(cfg.seed)
    print(cfg)
    if(save_files):
        create_folder(prep_dataset_path)
    # %%
    #Load dataset and dataset parameters
    raw_dataset_dict = call(cfg.dataset.load_function,_recursive_=False)
    data = raw_dataset_dict["train"]["data"] 
    labels = raw_dataset_dict["train"]["labels"] 
    test_data = raw_dataset_dict["test"]["data"]
    test_labels = raw_dataset_dict["test"]["labels"] 
    normal_class = raw_dataset_dict["params"]["normal_class"]
    leg = raw_dataset_dict["params"]["classes"]
    subplot_ecg(data,labels,num_to_plot=9,lead=0,indexes=range(9))

    if(dataset_split in ["all","all12"]):
        assert test_data is None
        assert test_labels is None
    if(save_files):
        with open(os.path.join(prep_dataset_path,"params"), 'wb') as file:
            pickle.dump(raw_dataset_dict["params"], file)
    # Generate dataset with FMM coefficients extraction
    print("Preprocessing data...")
    if(dataset_split.startswith("test")):
        data_in = [{"data":x,"label":y,"info":info} for x,y,info in zip(test_data,test_labels,raw_dataset_dict["test"]["info"])]
    elif(dataset_split.startswith("train") or  dataset_split.startswith("all")):
        data_in = [{"data":x,"label":y,"info":info} for x,y,info in zip(data,labels,raw_dataset_dict["train"]["info"])]
        if(cfg.dataset.name=="ptb"):
            #Delete a sample that gives problem for preprocessing
            to_delete_sample_index = 13411
            del data_in[to_delete_sample_index]
    with Pool(cpu_count()-4) as p:
        multi_process_results = p.map(preprocess_data_multi_core, data_in)
    beat_list = extract_beat_list(multi_process_results)
    print("Extracting coefficients...")
    with Pool(cpu_count()-4) as p:
        elapsed_times = p.map(save_fmm_parameters, beat_list) #It saves in prep_dataset_path
    elapsed_times = [item for item in elapsed_times if item is not None]
    elapsed_times= np.array(elapsed_times)
    np.save(os.path.join(prep_dataset_path,"elapsed_times.npy"), elapsed_times)

if __name__ == '__main__':
    my_app()