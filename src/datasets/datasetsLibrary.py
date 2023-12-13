#%%
import pandas as pd
import numpy as np
import os
import shutil
import datetime
from requests import head
import tensorflow as tf
import glob
from typing import List
import tqdm
import json
import wget
import wfdb
import ast
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from src.utils.general_functions import create_folder, get_download_file_name
import pickle
from src.utils.fmm import get_fmm_num_parameters, convert_fmm_dictionary_to_array,\
                            extract_fmm_lead_from_array, get_A_indexes, get_circular_indexes_as_boolean_t, \
                            get_fmm_num_parameters_circular, sort_fmm_coeffs_array
from src.utils.math import angle_vector_to_cos_sin
from sklearn.model_selection import train_test_split
import gdown
import zipfile
def str_to_date(instr:str, format:str)->datetime.date:
    return datetime.datetime.strptime(instr, format)

def kaggle_authenticate():
    #Setup to download from kaggle
    try:
        with open("kaggle.json") as f:
            kaggleCredentials = json.load(f)
    except Exception as e:
        raise FileNotFoundError("To download the dataset, please register to kaggle and download the token API kaggle.json")
    os.environ['KAGGLE_USERNAME'] = kaggleCredentials["username"]
    os.environ['KAGGLE_KEY'] = kaggleCredentials["key"]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api

def get_kaggle_dataset(id:str, filename:str, datapath:str):
    """Get a dataset from kaggle and unzip it

    Args:
        id (str): kaggle id
        filename (str): name of file to be saved
        datapath (str): data folder
    """
    api = kaggle_authenticate()
    api.dataset_download_files(id, path=datapath)
    fullFileName = os.path.join(datapath,filename)
    shutil.unpack_archive(filename=fullFileName,extract_dir=datapath)
    
def get_telecom_italy_dataset(datapath:str="./data"):
    zip_filename = "mobile-phone-activity.zip"
    fullpath = os.path.join(datapath,zip_filename)
    if(not(os.path.exists(fullpath))):
        get_kaggle_dataset(id='marcodena/mobile-phone-activity',
                            filename=zip_filename,
                            datapath=datapath)
    #TODO: handle not only calls, but also messages and geography
     
def get_khan_dataset(datapath:str="./data", onlyNormalAndCovid:bool=False,onlyCovidAndAll:bool=False,**kwargs):
    """
    Get ECG signals from khan hospital dataset (no images)
    https://www.kaggle.com/datasets/marcjuniornkengue/ecg-signal-covid
    """
    # classes=['Normal','HB','PMI', 'MI','COVID']
    fullpath = os.path.join(datapath,"ecg-signal-covid.zip")
    if(not(os.path.exists(fullpath))):
        get_kaggle_dataset(id='marcjuniornkengue/ecg-signal-covid',
                            filename="ecg-signal-covid.zip",
                            datapath=datapath)
    filepath = os.path.join(datapath,"covid_dataset_balanced.csv")
    df = pd.read_csv(filepath)
    data = df.iloc[:,:-1].values
    labels = df.iloc[:,-1].values
    assert int(onlyNormalAndCovid)+int(onlyCovidAndAll)<=1
    if(onlyNormalAndCovid):
        normalData = data[labels==0,:]
        covidData = data[labels==4,:]
        return normalData, covidData
    if(onlyCovidAndAll):
        covidData = data[labels==4,:]
        allOtherData = data[labels!=4,:]
        return covidData,allOtherData
    return (data, labels)
    
def get_swell_dataset(datapath:str="./data",**kwargs):
    outnameSwell = "swell-heart-rate-variability-hrv.zip"
    outpathSwell = os.path.join(datapath,outnameSwell)
    api = kaggle_authenticate()
    api.dataset_download_files('qiriro/swell-heart-rate-variability-hrv', path=datapath)
    shutil.unpack_archive(filename=outpathSwell,extract_dir=datapath)
    swellTrainfile = os.path.join(datapath, "hrv dataset", "data", "raw","labels",  "hrv stress labels.xlsx")
    dfDict = pd.read_excel(swellTrainfile, sheet_name=None)
    swellBpmSequencesList= [] 
    for id,clientDf in dfDict.items():
        bpdf = pd.to_numeric(clientDf["HR"], errors='coerce')
        bpnumpy = bpdf.to_numpy()
        bpnumpy = bpnumpy[np.logical_not(np.isnan(bpnumpy))]
        swellBpmSequencesList.append(bpnumpy)
    swellBpmSequencesList = [x for x in swellBpmSequencesList if np.size(x)!=0]
    return swellBpmSequencesList

def get_mishra_covid_dataset(datapath:str="./data",**kwargs):
    """ 
    Covid dataset from:
    Mishra, T., Wang, M., Metwally, A.A. et al. Pre-symptomatic detection of COVID-19 from smartwatch data. Nat Biomed Eng 4, 1208–1220 (2020). https://doi.org/10.1038/s41551-020-00640-6
    """
    outnameMishra = "COVID-19-Wearables.zip"
    outpathMishra = os.path.join(datapath,outnameMishra)
    if(not(os.path.exists(outpathMishra))):
        urlMishra="https://storage.googleapis.com/gbsc-gcp-project-ipop_public/COVID-19/COVID-19-Wearables.zip"
        wget.download(urlMishra,outpathMishra)
    shutil.unpack_archive(filename=outpathMishra,extract_dir=datapath)
    def get_bpm_from_csv(filepath:str)->np.ndarray:
        df = pd.read_csv(filepath)
        df = df[["datetime","heartrate"]]
        dateformat='%Y-%m-%d %H:%M:%S'
        df['datetime'] = df['datetime'].map(lambda x: str_to_date(x,format=dateformat)) #convert string to datetime
        df = df.resample('T',on='datetime').mean() #average per minute
        return df["heartrate"].to_numpy()
    def get_bpm_from_directory(dirpath:str)->List[tf.data.Dataset]:
        pattern = os.path.join(dirpath,"*_hr.csv")
        csvList = glob.glob(pattern)
        # print(csvList)
        # bpmSequencesList = [get_bpm_from_csv(file) for file in csvList]
        bpmSequencesList = []
        for file in tqdm.tqdm(csvList):
            bpnumpy = get_bpm_from_csv(file)
            bpnumpy = bpnumpy[np.logical_not(np.isnan(bpnumpy))]
            bpmSequencesList.append(bpnumpy)
        return bpmSequencesList
    mishraBpmSequencesList = get_bpm_from_directory(os.path.join(datapath,"COVID-19-Wearables"))
    return mishraBpmSequencesList

def get_ecg5000_dataset(datapath:str="./data",**kwargs):
    # Get ecg 5000 dataset from "http://timeseriesclassification.com/Downloads/ECG5000.zip"
    # train has 500 samples, test 4500. 5 classes
    # Return a dictionary for test and train data and labels. Output
    # dataset_dict["train"]["data"] or ecg5000_dict["train"]["labels"] 
    # Same for test
    outname_ecg5000 = "ECG5000.zip"
    outpath_ecg5000 = os.path.join(datapath,outname_ecg5000)
    if(not(os.path.exists(outpath_ecg5000))):
        ecg5000_url = "http://timeseriesclassification.com/aeon-toolkit/ECG5000.zip"
        wget.download(ecg5000_url,outpath_ecg5000)
    shutil.unpack_archive(filename=outpath_ecg5000,extract_dir=datapath)
    dataset_dict = {}
    for datatype in ["TRAIN", "TEST"]:
        filepath = os.path.join(datapath,f"ECG5000_{datatype}.txt")
        ecg = np.loadtxt(filepath)
        ecg_labels = ecg[:,0] - 1 # Decrease by one to have 0-4 classes labels instead of 1-5
        ecg_data = ecg[:,1:]
        ecg_data = np.expand_dims(ecg_data, axis=-1).astype(np.float32)
        ecg_labels = ecg_labels.astype(int)
        ecg_data_shape = np.array([ecg_data.shape[1] for _ in ecg_data])
        dataset_dict[str.lower(datatype)] = {"data":ecg_data,"labels":ecg_labels, "sizes":ecg_data_shape}
    dataset_dict["params"]={"normal_class":0,"classes":["Normal", "Supraventricular", "Ventricular", "Fusion", "Unclassifiable"]}
    return dataset_dict

def get_sweden_regional_covid_statistics(datapath:str="./data"):
    outname = "statistik-covid19-vardforlopp.xlsx"
    outpath = os.path.join(datapath,outname)
    lan_name = "Län"
    column_names = [lan_name,"Total cases","Death rate", "Not intensive care", "Not intensive care (%)","Intensive care", "Intensive care (%)"]
    if(not(os.path.exists(outpath))):
        url_data = "https://www.socialstyrelsen.se/globalassets/sharepoint-dokument/dokument-webb/statistik/statistik-covid19-vardforlopp.xlsx"
        wget.download(url_data,outpath)
    def get_number_covid_patients_from_xls(filepath:str)->np.ndarray:
        # Get number of reported patients in hospitals (including ICU) for more than 30 days in Sweden per län
        df = pd.read_excel(filepath,sheet_name="Slutenvårdade per region",skiprows=10,skipfooter=2,
                        header=None, names = column_names)
        num_rows = df.count(axis=0)[column_names[0]]
        assert num_rows == 21 # There are 21 läns in Sweddn
        df = df[[column_names[0],column_names[1]]] # Get only län and number of hospitalized patients
        return df
    covid_df = get_number_covid_patients_from_xls(outpath)
    # Compute region weight based on number of hospitalized covid cases
    weight_col_name = "covid_weight"
    covid_df[weight_col_name] = covid_df[[column_names[1]]] / covid_df[[column_names[1]]].sum() #df.sum(df[[column_names[1]]])
    # print(covid_df.head(100))
    assert covid_df[weight_col_name].sum() ==1.0

    # #Use this to automatically download population statistics in Sweden
    # import requests
    # import json
    # # from pyscbwrapper import SCB
    # # scb = SCB("sv")
    # sweden_population_statistics_url = "https://api.scb.se/OV0104/v1/doris/en/ssd/START/BE/BE0101/BE0101A/FolkmangdNov"
    # json_request_path = os.path.join(datapath,"population_register_request.json")
    # with open(json_request_path,"r") as f:
    #     query = json.load(f)
    # import requests
    # session = requests.Session()
    # response = session.post(sweden_population_statistics_url, json=query)
    # # response_json = json.loads(response.content.decode('utf-8-sig'))

    #Use this to load static file
    outname = "sweden_population.xlsx"
    outpath = os.path.join(datapath,outname)
    statistics_column_names = [lan_name,"Population"]
    population_df = pd.read_excel(outpath,skiprows=3,skipfooter=41,header=None, names = statistics_column_names)
    population_df[lan_name] = population_df[lan_name].map(lambda x: x.split()[1]) # TODO: do split accoring to regex instead of split
    population_df[lan_name] = population_df[lan_name].map(lambda x:  "Västra Götaland" if(x=="Västra")else x) #When using split, län of two words have problems
    weight_col_name = "population_weight"
    population_df[weight_col_name] = population_df[[statistics_column_names[1]]] / population_df[[statistics_column_names[1]]].sum()

    df = pd.merge(population_df,covid_df,on=lan_name)
    # print(df.head(100))
    # covid_df[lan_name].astype(str)
    # population_df[lan_name].astype(str)
    # population_df.set_index(lan_name).join(population_df.set_index(lan_name))
    # population_df.join(covid_df,on=lan_name)
    # print(population_df.head(100))
    return df

def get_mit_bih_dataset(datapath:str="./data",lead:int=None,**kwargs):
    filename = "mit_bih.zip"
    outpath = os.path.join(datapath,filename)
    folder_name = os.path.join(datapath,"mit_bih")
    url = "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip" #https://physionet.org/files/mitdb/1.0.0/
    if(not(os.path.exists(outpath))):
        wget.download(url,outpath)
        shutil.unpack_archive(filename=outpath,extract_dir=datapath)
        os.rename(os.path.join(datapath,"mit-bih-arrhythmia-database-1.0.0"), folder_name)
    path = folder_name
    # with open(os.path.join(path,"RECORDS"),"r") as f:
    #     record_names = f.read()
    # record_names = record_names.splitlines()
    # record_names.remove("114") #This record is bad because of leads swap
    record_len = 650000
    # num_samples = len(record_names)
    num_leads = 2
    # ecg_signals = np.zeros_like(num_samples,record_len,num_leads) 
    # annotation_list = []
    # abnormal_classes = ["S", "V", "F", "Q"]
    ds1_samples = [101,106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201,
    203, 205, 207, 208, 209, 215, 220, 223, 230]
    ds2_samples = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
        213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
    def mitbih_map_labels(x):
        if(x=="N"):
            return 0
        elif(x in ['L','R','V','/','A','f','F','j','a','E','J','e','S']):
            return 1
        else:
            return -1
    def mit_bih_get_samples(samples_list):
        num_samples = len(samples_list)
        ecg_signals = np.zeros((num_samples,record_len,num_leads)) 
        annotation_list = []
        for i,record_name in enumerate(samples_list):
            record = wfdb.rdrecord(os.path.join(path,str(record_name)))
            annotation = wfdb.rdann(os.path.join(path,str(record_name)), 'atr')
            # do something with the ECG signals and annotation
            ecg_signals[i,:,:] = record.p_signal
            # labels = np.array(annotation.symbol)!="N" # Use only normal/abnormal labels
            labels = np.vectorize(mitbih_map_labels)(np.array(annotation.symbol))
            annotation_list.append({"labels":labels,"peaks":annotation.sample})
        return ecg_signals,annotation_list
    # data = [wfdb.rdsamp(os.path.join(path,str(f))) for f in range(100,101)] #df.filename_lr
    normal_class = 0 
    leg = ["Normal", "Abnormal"]
    out_params = {"classes":leg,"normal_class":normal_class}
    out_dict = {}
    ecg_signals_train,annotation_list_train = mit_bih_get_samples(ds1_samples)
    ecg_signals_test,annotation_list_test = mit_bih_get_samples(ds2_samples)
    lead_is_none = lead is None or lead=="None"
    if(not(lead_is_none)):
        ecg_signals_train = ecg_signals_train[:,:,lead]
        ecg_signals_test = ecg_signals_test[:,:,lead]

    while(ecg_signals_train.ndim<3): # Expand dimensions until we get 3 dimensions (sample, )
        ecg_signals_train = np.expand_dims(ecg_signals_train, axis=-1).astype(np.float32)
    while(ecg_signals_test.ndim<3): # Expand dimensions until we get 3 dimensions (sample, )
        ecg_signals_test = np.expand_dims(ecg_signals_test, axis=-1).astype(np.float32)
    out_dict["train"] = {"data":ecg_signals_train,"labels":None,"annotations":annotation_list_train}
    out_dict["test"] = {"data":ecg_signals_test,"labels":None,"annotations":annotation_list_test}
    # out_params["train_annotations"] = annotation_list_train
    # out_params["test_annotations"] = annotation_list_test
    out_dict["params"] = out_params
    return out_dict

def get_shaoxing_dataset(datapath:str="./data", lead:int=None, test_size:float=0.2, split_seed:int=None ):
    assert split_seed is not None
    data_subdir = os.path.join(datapath,"shaoxing")
    create_folder(data_subdir) # Create shaoxing folder if not present
    common_url = "https://figshare.com/ndownloader/files/"
    web_resources_dict = {15651296:'RhythmNames.xlsx', #https://figshare.com/ndownloader/files/15651296
                          15651293:'ConditionNames.xlsx', #https://figshare.com/ndownloader/files/15651293
                          15653771:'Diagnostics.xlsx', #https://figshare.com/ndownloader/files/15653771
                          15653762: 'AttributesDictionary.xlsx', #https://figshare.com/ndownloader/files/15653762
                          15651326:'ECGData.zip', #https://figshare.com/ndownloader/files/15651326
                          15652862:'ECGDataDenoised.zip' #https://figshare.com/ndownloader/files/15652862
        
    }
    file_paths = []
    for id,download_file_name in web_resources_dict.items(): #Download all metadata and data
        url = common_url + str(id)
        download_file_name = os.path.join(data_subdir,download_file_name)
        file_paths.append(download_file_name) # Save paths of downloaded files 
        # Download only if not already present in the subfolder
        if(not(os.path.exists(download_file_name))): 
            wget.download(url,data_subdir)
        filepath = Path(download_file_name)
        unzipped_filepath = os.path.join(filepath.parent,filepath.stem)
        # unzip file if it is zip and it is not already unzipped
        if(filepath.suffix == '.zip' and not(os.path.exists(unzipped_filepath))): 
            print(f"Unzipping file {download_file_name}...")
            shutil.unpack_archive(filename=download_file_name,extract_dir=data_subdir)
    # Get file names from  Diagnostics.xlsx
    diagnostic_index = 2 # Index in file_paths (same in dict) is 2 for diagnostic
    diagnostics_df = pd.read_excel(file_paths[diagnostic_index])
    # Get labels and other useful metadata from Diagnostics.xlsx
    # diagnostics_df = pd.read_excel("/home/giacomo/ScalableFederatedLearning/data/shaoxing/Diagnostics.xlsx")
    le = LabelEncoder()
    diagnostics_df["categorical_rhythm"] =le.fit_transform(diagnostics_df["Rhythm"])
    mlb = MultiLabelBinarizer()
    diagnostics_df['Beat'] = diagnostics_df.Beat.apply(lambda x: x.split(' '))
    diagnostics_df["binary_beat_label"] = mlb.fit_transform(diagnostics_df["Beat"]).tolist()
    diagnostics_df["categorical_beat"] = diagnostics_df["binary_beat_label"].apply(lambda x: np.reshape(np.argwhere(x),(-1,)))
    diagnostics_df.set_index('FileName', inplace=True)

    # Get data
    ecg_denoised_path = os.path.join(data_subdir,"ECGDataDenoised")
    files_list = glob.glob(os.path.join(ecg_denoised_path , "*.csv"))
    files_list.pop(1701) #This sample has not enough data
    files_list.pop(8815)
    num_timesteps = 5000
    num_features = 12
    ecg_data = np.zeros((len(files_list),num_timesteps,num_features))
    ecg_labels = np.zeros((len(files_list),),dtype=int)
    for i,myfile in enumerate(tqdm.tqdm(files_list)):
        file_index = Path(myfile).stem
        df = pd.read_csv(myfile,header=None)
        df_numpy = df.to_numpy(dtype=np.float32)
        ecg_data[i,:,:] = df_numpy
        sample_label = diagnostics_df.loc[file_index]["categorical_beat"]
        if(sample_label.shape[0]!=1): # If we have different than 1 label set label to -1 and remove it after
            ecg_labels[i] = -1
        else:
            ecg_labels[i] = sample_label
    # Delete negative labels (correspondent to multilabels)
    wrong_labels_indexes = np.where(ecg_labels==-1)[0]
    ecg_data = np.delete(arr=ecg_data,obj=wrong_labels_indexes,axis=0)
    ecg_labels = np.delete(arr=ecg_labels,obj=wrong_labels_indexes,axis=0)
    
    lead_is_none = lead is None or lead=="None"
    if(not(lead_is_none)):
        ecg_signals_train = ecg_signals_train[:,:,lead]
        ecg_signals_test = ecg_signals_test[:,:,lead]
    from src.utils.preprocessing import normalize_single_rows
    ecg_data = normalize_single_rows(ecg_data)
    if(test_size>0.0):
        X_train,X_test,y_train,y_test = train_test_split(ecg_data,ecg_labels,
                                                     test_size=test_size,
                                                     random_state=split_seed)
    else: 
        X_train,X_test,y_train,y_test = ecg_data,None,ecg_labels,None
    leg = mlb.classes_.tolist() 
    class_tuple_list = []
    for class_name in mlb.classes_:
        class_id = mlb.transform([[class_name]])
        class_id = np.where(class_id)[1]
        class_tuple_list.append((class_id,class_name))
    class_tuple_list.sort()
    leg = list(list(zip(*class_tuple_list))[1])
    normal_class = int(np.where(mlb.transform([["NONE"]]))[1])
    out_params = {"classes":leg,"normal_class":normal_class}
    out_dict = {}
    out_dict["train"] = {"data":X_train,"labels":y_train}
    out_dict["test"] = {"data":X_test,"labels":y_test}
    out_dict["params"] = out_params 
    return out_dict

def get_shaoxing_fmm_dataset(datapath:str="./data", frequency:int=500,lead:int=None, 
                             test_size:float=0.2, split_seed:int=None ,**kwargs):
    datapath_folder = os.path.join(datapath, "shaoxing_fmm")
    if(not(os.path.exists(datapath_folder))):
        os.makedirs(datapath_folder)
        file_id = '1FjvmVb8-PnpDdoBwv-Eqb89tYFOCx9KW'
        url = f'https://drive.google.com/uc?id={file_id}'
        zipfile_path = os.path.join(datapath_folder, "shaoxing_fmm.zip") 
        gdown.download(url, zipfile_path, quiet=False)
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(datapath_folder)
    num_waves=kwargs["num_waves"]
    num_leads=kwargs["num_leads"]
    num_parameters_per_ecg,_ = get_fmm_num_parameters(num_leads=num_leads,num_waves=num_waves)
    num_parameters_per_ecg_circular,_ = get_fmm_num_parameters_circular(num_leads=num_leads,num_waves=num_waves)
    circular_indexes = get_circular_indexes_as_boolean_t(num_leads=num_leads,num_waves=num_waves)
    sequence_len = kwargs["sequence_length"]
    def get_numpy_dataset_from_slice(slice:str):
        # Slice can be "train" of "test"
        # Get dataset from list of files in slice path
        print(f"Loading \"{slice}\" folder")
        samples_path = os.path.join(datapath_folder,slice)
        samples_files = os.listdir(samples_path)
        samples_files.remove("params")
        num_samples = len(samples_files)
        data = np.zeros((num_samples,sequence_len,num_leads))
        labels = -np.ones((num_samples),dtype=int)
        sizes = -np.ones((num_samples))
        coefficients = np.zeros((num_samples,num_parameters_per_ecg))
        # coefficients_circular = np.zeros((num_samples,num_parameters_per_ecg_circular))
        for row_number,file_name in tqdm.tqdm(enumerate(samples_files),total=len(samples_files)):
            full_file_name = os.path.join(samples_path,file_name)
            split_file_name = file_name.replace("_",".").split(".")
            sample_id, beat_id = int(split_file_name[1]),int(split_file_name[3])
            with open(full_file_name, 'rb') as f:
                sample = pickle.load(file=f)
            sample["sample_id"] = sample_id
            sample["beat_id"] = beat_id
            sample_len = sample["len"]
            sample_sequence = sample["data"] if num_leads==12 else sample["data"][:,lead]
            while(sample_sequence.ndim<2): # Expand dimensions until we get 2 dimensions
                sample_sequence = np.expand_dims(sample_sequence, axis=-1).astype(np.float32)
            if(sample_len>sequence_len):
                data[row_number,:,:]= sample_sequence[:sequence_len,:]
            elif(sample_len<=sequence_len):
                data[row_number,:sample_len-1,:] = sample_sequence
            labels[row_number] = int(sample["label"])
            sizes[row_number] = int(sample["len"])
            sample_coefficients = convert_fmm_dictionary_to_array(sample["coefficients"])
            try:
                sample_coefficients = sample_coefficients if num_leads==12 else \
                                extract_fmm_lead_from_array(fmm_coefficients_array=sample_coefficients, 
                                                            lead=lead,num_leads=8,num_waves=num_waves) # Fmm preprocessing from R code produces 8 leads
            except:
                print(f"Removing sample {sample_id} beat {beat_id} due to bad format")
                labels[row_number]=-1
                continue
            coefficients[row_number] = sample_coefficients
            # coefficients_circular[row_number] = angle_vector_to_cos_sin(sample_coefficients,circular_indexes)
        # Before returning, delete rows which have too high A parameter
        # a_indexes = [get_A_indexes(wave_index=i,num_leads=num_leads,num_waves=num_waves)[0] for i in range(num_waves)]
        # to_delete_rows = np.where(coefficients[:,a_indexes]>5)[0]
        # data = np.delete(arr=data,obj=to_delete_rows,axis=0)
        # labels = np.delete(arr=labels,obj=to_delete_rows,axis=0)
        # coefficients = np.delete(arr=coefficients,obj=to_delete_rows,axis=0)
        # sizes = np.delete(arr=sizes,obj=to_delete_rows,axis=0)
        #Delete rows with negative label
        to_delete_rows = np.where(labels<0.0)[0]
        data = np.delete(arr=data,obj=to_delete_rows,axis=0)
        labels = np.delete(arr=labels,obj=to_delete_rows,axis=0)
        coefficients = np.delete(arr=coefficients,obj=to_delete_rows,axis=0)
        sizes = np.delete(arr=sizes,obj=to_delete_rows,axis=0)
        return data,labels, coefficients,sizes
    # X_train, y_train, coeffs_train,sizes_train = get_numpy_dataset_from_slice(slice="train")
    ds_slice = "all"
    X_test, y_test, coeffs_test,sizes_test = get_numpy_dataset_from_slice(slice=ds_slice) # usd test2 for simple dataset
    X_train,X_test, y_train, y_test, coeffs_train, coeffs_test,sizes_train,sizes_test = \
        train_test_split(X_test, y_test, coeffs_test,sizes_test,
                            test_size=test_size,
                            random_state=split_seed)
    coeffs_train = sort_fmm_coeffs_array(fmm_array=coeffs_train,num_leads=num_leads,num_waves=num_waves)
    coeffs_test = sort_fmm_coeffs_array(fmm_array=coeffs_test,num_leads=num_leads,num_waves=num_waves)
    circular_indexes = np.squeeze(np.argwhere(circular_indexes))
    coeffs_train_ang = angle_vector_to_cos_sin(coeffs_train,ang_indexes=circular_indexes,zero_one_interval=True)
    coeffs_test_ang = angle_vector_to_cos_sin(coeffs_test,ang_indexes=circular_indexes,zero_one_interval=True)
    # leg = np.unique(y_train).tolist()
    # out_params = {"classes":leg,"normal_class":36}
    out_dict = {}
    out_dict["train"] = {"data":X_train,"labels":y_train,"coefficients":coeffs_train,"coefficients_ang":coeffs_train_ang,"sizes":sizes_train}
    out_dict["test"] = {"data":X_test,"labels":y_test,"coefficients":coeffs_test,"coefficients_ang":coeffs_test_ang, "sizes":sizes_test}
    with open(os.path.join(datapath_folder,ds_slice,"params"), 'rb') as f:
        out_dict["params"] = pickle.load(file=f)
    return out_dict

def get_ptb_xl_fmm_dataset(datapath:str="./data", frequency:int=100,lead:int=None, delete_high_A=True, **kwargs):
    # test_path = os.path.join(datapath,"test")
    # num_leads = 12 if (lead is None or lead=="None") else len(lead)
    datapath_folder = os.path.join(datapath, "ptb_xl_fmm")
    if(not(os.path.exists(datapath_folder))):
        os.makedirs(datapath_folder)
        file_id = '1nYRvbVYJJXPbCwKEqOIeCLnMXdIDAka7'
        url = f'https://drive.google.com/uc?id={file_id}'
        zipfile_path = os.path.join(datapath_folder, "ptb_xl_fmm.zip") 
        gdown.download(url, zipfile_path, quiet=False)
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(datapath_folder)
    num_waves=kwargs["num_waves"]
    num_leads=kwargs["num_leads"]
    num_parameters_per_ecg,_ = get_fmm_num_parameters(num_leads=num_leads,num_waves=num_waves)
    num_parameters_per_ecg_circular,_ = get_fmm_num_parameters_circular(num_leads=num_leads,num_waves=num_waves)
    circular_indexes = get_circular_indexes_as_boolean_t(num_leads=num_leads,num_waves=num_waves)
    sequence_len = kwargs["sequence_length"]
    def get_numpy_dataset_from_slice(slice:str):
        # Slice can be "train" of "test"
        # Get dataset from list of files in slice path
        print(f"Loading \"{slice}\" folder")
        samples_path = os.path.join(datapath_folder,slice)
        samples_files = os.listdir(samples_path)
        for to_del_name in ["params","elapsed_times","elapsed_times.npy"]:
            try:
                samples_files.remove(to_del_name)
            except:
                pass
        with open(os.path.join(samples_path,samples_files[0]), 'rb') as f:
            sample = pickle.load(file=f)
            sample_keys = list(sample.keys())
        output_dict = {}
        num_samples = len(samples_files)
        for key in sample_keys:
            output_dict[key] = -np.ones((num_samples))
        output_dict["label"] = -np.ones((num_samples),dtype=int)
        output_dict["data"] = np.zeros((num_samples,sequence_len,num_leads))
        output_dict["coefficients"] = np.zeros((num_samples,num_parameters_per_ecg))
        output_dict["predicted_coefficients"] = np.zeros((num_samples,num_parameters_per_ecg))
        output_dict["filenames"] = np.zeros((num_samples),dtype="U30")
        # data = np.zeros((num_samples,sequence_len,num_leads))
        # labels = -np.ones((num_samples),dtype=int)
        # sizes = -np.ones((num_samples))
        # coefficients = np.zeros((num_samples,num_parameters_per_ecg))
        
        # coefficients_circular = np.zeros((num_samples,num_parameters_per_ecg_circular))
        for row_number,file_name in tqdm.tqdm(enumerate(samples_files),total=len(samples_files)):
            full_file_name = os.path.join(samples_path,file_name)
            split_file_name = file_name.replace("_",".").split(".")
            sample_id, beat_id = int(split_file_name[1]),int(split_file_name[3])
            with open(full_file_name, 'rb') as f:
                sample = pickle.load(file=f)
            # sample_keys = sample.keys()
            # sample_keys = ['coefficients', 'data', 'label', 'peak_index', 'len', 'sample_index', 'beat_index', 'ecg_id', 'patient_id', 'age', 'sex', 'height', 'weight']
            sample_len = sample["len"]
            sample_sequence = sample["data"] if num_leads==12 else sample["data"][:,lead]
            while(sample_sequence.ndim<2): # Expand dimensions until we get 2 dimensions
                sample_sequence = np.expand_dims(sample_sequence, axis=-1).astype(np.float32)
            if(sample_len>sequence_len):
                output_dict["data"][row_number,:,:] = sample_sequence[:sequence_len,:]
            elif(sample_len<=sequence_len):
                output_dict["data"][row_number,:sample_len-1,:] = sample_sequence
            output_dict["label"][row_number] = int(sample["label"])
            output_dict["len"][row_number] = int(sample["len"])
            output_dict["filenames"][row_number] = file_name
            sample_coefficients = convert_fmm_dictionary_to_array(sample["coefficients"])
            # predicted_sample_coefficients = convert_fmm_dictionary_to_array(sample["predicted_coefficients"])
            try:
                sample_coefficients = sample_coefficients if num_leads==12 else \
                                extract_fmm_lead_from_array(fmm_coefficients_array=sample_coefficients, 
                                                            lead=lead,num_leads=8,num_waves=num_waves) # Fmm preprocessing from R code produces 8 leads
                # predicted_sample_coefficients = predicted_sample_coefficients if num_leads==12 else \
                #                 extract_fmm_lead_from_array(fmm_coefficients_array=predicted_sample_coefficients,
                #                                                           lead=lead,num_leads=8,num_waves=num_waves)
                output_dict["coefficients"][row_number] = sample_coefficients
                # output_dict["predicted_coefficients"][row_number] = predicted_sample_coefficients
            except:
                print(full_file_name)
                raise(ValueError)
            # coefficients_circular[row_number] = angle_vector_to_cos_sin(sample_coefficients,circular_indexes)
        # Before returning, delete rows which have too high A parameter
        if(delete_high_A):
            for n in range(num_leads):
                a_indexes = [get_A_indexes(wave_index=i,num_leads=num_leads,num_waves=num_waves)[0]+n for i in range(num_waves)]
                to_delete_rows = np.where(output_dict["coefficients"][:,a_indexes]>5)[0]
                for key in sample_keys:
                    output_dict[key] = np.delete(arr=output_dict[key],obj=to_delete_rows,axis=0)
        output_dict["labels"] = output_dict.pop("label")
        output_dict["sizes"] = output_dict.pop("len")
        return output_dict
    train_folder_name = "train" # Modify to train12_predicted and test12_predicted for 12 leads instead of 8
    test_folder_name = "test"
    train_dict  = get_numpy_dataset_from_slice(slice=train_folder_name) 
    test_dict = get_numpy_dataset_from_slice(slice=test_folder_name) 
    
    coeffs_train = sort_fmm_coeffs_array(fmm_array=train_dict["coefficients"], num_leads=num_leads, num_waves=num_waves)
    coeffs_test = sort_fmm_coeffs_array(fmm_array=test_dict["coefficients"], num_leads=num_leads, num_waves=num_waves)
    circular_indexes = np.squeeze(np.argwhere(circular_indexes))
    train_dict["coefficients_ang"] = angle_vector_to_cos_sin(coeffs_train, ang_indexes=circular_indexes, zero_one_interval=True)
    test_dict["coefficients_ang"] = angle_vector_to_cos_sin(coeffs_test, ang_indexes=circular_indexes, zero_one_interval=True)
    # leg = np.unique(y_train).tolist()
    # out_params = {"classes":leg,"normal_class":3}
    out_dict = {}
    # out_dict["train"] = {"data":X_train,"labels":y_train}
    # out_dict["test"] = {"data":X_test,"labels":y_test}
    out_dict["train"] = train_dict #{"data":X_train,"labels":y_train,"coefficients":coeffs_train,"coefficients_ang":coeffs_train_ang,"sizes":sizes_train,"filenames":train_files}
    out_dict["test"] = test_dict #{"data":X_test,"labels":y_test,"coefficients":coeffs_test,"coefficients_ang":coeffs_test_ang, "sizes":sizes_test,"filenames":test_files}
    # out_dict["params"] = out_params 
    with open(os.path.join(datapath_folder,train_folder_name,"params"), 'rb') as f:
        out_dict["params"] = pickle.load(file=f)
    return out_dict

    
# wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
def get_ptb_xl_dataset(datapath:str="./data", frequency:int=100, delete_multi_label:bool=True,lead:int=None,**kwargs):
    """
    Get ECG signals from ptb dataset. (2020)
    Sampling frequency of 100 Hz or 500 Hz are available
    https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2.zip

    Or in alternative from kaggle:
    https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset
    """

    filename = "ptb_xl.zip"
    outpath = os.path.join(datapath,filename)
    folder_name = os.path.join(datapath,"ptb_xl")
    url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    if(not(os.path.exists(outpath))):
        wget.download(url,outpath)
        shutil.unpack_archive(filename=outpath,extract_dir=datapath)
        os.rename(os.path.join(datapath,"ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"), folder_name)
    def load_raw_data(df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data
    path = folder_name+"/"
    sampling_rate=frequency
    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)
    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
    # Split data into train and test
    test_fold = 10
    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train_str = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    
    # for key_name in ["ecg_id","patient_id","age","sex","height","weight"]:
    info_keys = ["ecg_id","patient_id","age","sex","height","weight"]
    Y["ecg_id"] = Y.index
    train_info_list = Y[(Y.strat_fold != test_fold)][info_keys].to_dict(orient='records')
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test_str = Y[Y.strat_fold == test_fold].diagnostic_superclass
    test_info_list = Y[(Y.strat_fold == test_fold)][info_keys].to_dict(orient='records')
    def delete_multiple_labels(x,y,info_list,mlb):
        # Delete multiple label and no label entries from y and (correspondently) in x. 
        # Also, binarize the labels
        y_bin = mlb.transform(y)
        y_sums = np.sum(np.abs(y_bin), axis=1)
        to_delete_rows = np.where(np.logical_or((y_sums==0),(y_sums>1)))[0]
        kept_rows = np.setdiff1d(np.arange(y_sums.shape[0]), to_delete_rows, assume_unique=False)
        y_reduced = np.delete(y_bin,to_delete_rows, axis=0) 
        x_reduced = np.delete(x,to_delete_rows, axis=0)
        info_list_reduced = [item for index, item in enumerate(info_list) if index not in to_delete_rows]
        return x_reduced,y_reduced,info_list_reduced,kept_rows
        
    if(delete_multi_label):
        mlb = MultiLabelBinarizer()
        mlb.fit(y_train_str)
        # Delete 0 and multi-label samples:
        X_train, y_train,train_info_list,kept_rows_train = delete_multiple_labels(X_train, y_train_str,train_info_list,mlb)
        X_test, y_test,test_info_list,kept_rows_test = delete_multiple_labels(X_test, y_test_str,test_info_list,mlb)
        
        assert X_train.shape[0]==y_train.shape[0] and X_train.shape[0]==len(train_info_list)
        assert X_test.shape[0]==y_test.shape[0] and X_test.shape[0]==len(test_info_list)
        y_train = np.argwhere(y_train)[:,1]
        y_test = np.argwhere(y_test)[:,1]
    lead_is_none = lead is None or lead=="None"
    if(not(lead_is_none)):
        X_train = X_train[:,:,lead]
        X_test = X_test[:,:,lead]
    elif(kwargs.get("hexad",None)==True):
        # Use heaxad system for lead selection
        # All leads in order: (0I, 1II, 2III, 3AVL, 4AVR, 5AVF, 6V1, 7V2, 8V3, 9V4, 10V5, 11V6)
        # 0,1,2,3,4,5,7,9
        # V1, V3, V5, V6 to be deleted
        hexad_leads = np.array([0,1,2,3,4,5,7,9])
        X_train = X_train[:,:,hexad_leads]
        X_test = X_test[:,:,hexad_leads]
    leg=(mlb.classes_).tolist()
    normal_class = np.argwhere(mlb.classes_=="NORM")[0][0]
    out_params = {"classes":leg,"normal_class":normal_class}
    out_dict = {}
    while(X_train.ndim<3): # Expand dimensions until we get 3 dimensions (sample, )
        X_train = np.expand_dims(X_train, axis=-1).astype(np.float32)
    while(X_test.ndim<3): # Expand dimensions until we get 3 dimensions (sample, )
        X_test = np.expand_dims(X_test, axis=-1).astype(np.float32)
    out_dict["train"] = {"data":X_train,"labels":y_train,"info":train_info_list}
    out_dict["test"] = {"data":X_test,"labels":y_test,"info":test_info_list}
    out_dict["params"] = out_params
    return out_dict

#%%
if __name__ == '__main__':
    pass