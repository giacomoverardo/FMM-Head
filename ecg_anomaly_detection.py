from silence_tensorflow import silence_tensorflow
import os
import numpy as np
from keras import backend as K
import logging, os
import tensorflow as tf
from tensorflow.data import Dataset
from src.plot.general import plot_scalar_dictionary
from src.plot.series import *
from src.plot.vaeplot import *
from src.plot.fmm import plot_fmm_wave_from_coefficients
from src.utils.metrics import *
from src.utils.math import *
from src.utils.general_functions import *
from src.utils.preprocessing import *
from src.utils.nn import *
from src.utils.callbacks import *
from src.utils.fmm import get_waves_from_fmm_model,sort_fmm_coeffs_array, get_parameters_names_list, \
                        get_beta_indexes, get_M_indexes, get_A_indexes,get_alpha_indexes, get_omega_indexes, \
                        get_circular_indexes_as_boolean_t, generate_wave, convert_fmm_array_to_dict, \
                        reconstruct_FMM_leads_from_FMM_array, get_loss_from_fmm_model, expand_fmm_scalar_coefficients
import hydra
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from hydra.utils import get_original_cwd, instantiate, call
import logging 
from omegaconf import OmegaConf
import time
import tqdm
import pickle

logger = logging.getLogger(__name__)
os.environ["HYDRA_FULL_ERROR"] = "1"
@hydra.main(version_base="1.3", config_path="conf", config_name="ecg_anomaly_detection") 
def my_app(cfg) -> None:
    logger.info("Starting ecg anomaly detection script")
    # tf.sysconfig.get_build_info()["cuda_version"]
    set_ml_seeds(cfg.seed)
    print(cfg)

    # Save configuration file
    create_folder(cfg.tb_output_dir)
    filename = os.path.join(cfg.tb_output_dir,f"conf.yaml")
    with open(filename,"w") as fp:
        OmegaConf.save(config=cfg, f=fp)
    #Image saving function
    def save_png_eps_figure(filename):
        if(cfg.save_plots):
            full_filename = os.path.join(cfg.tb_output_dir,filename)
            plt.savefig(full_filename+".png")
            plt.savefig(full_filename+".eps")
    #Dict saving function
    def save_dict(in_dict, filename):
        if(cfg.save_plots):
            full_filename = os.path.join(cfg.tb_output_dir,filename)
            save_dict_to_binary(in_dict,full_filename)
    #Numpy saving function
    def save_np(np_array, filename):
        if(cfg.save_plots):
            full_filename = os.path.join(cfg.tb_output_dir,filename)
            np.save(full_filename, np_array)
    # Model names
    fmm_model_names = ["fmm_bert_ecg","fmm_cae", "fmm_encdec_ad", "fmm_lstm_ae", "fmm_ecgnet", "fmm_dense_ae"]
    non_ae_baseline_models = ["diffusion_ae" ,"ecg_adgan"]

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
    save_png_eps_figure("inputs")

    # %%
    # Plot some example time series
    sample_index_list = np.random.randint(0,raw_dataset_dict["train"]["data"].shape[0],(9))
    if(cfg.dataset.name in ["ptb_xl_fmm","shaoxing_fmm"]):
        lead = 0
        for sample_index in sample_index_list:
            original_seq = raw_dataset_dict["train"]["data"][sample_index] 
            fmm_coeff_array = raw_dataset_dict["train"]["coefficients"][sample_index] 
            seq_len = int(raw_dataset_dict["train"]["sizes"][sample_index])
            seq_label = leg[raw_dataset_dict["train"]["labels"][sample_index]]
            plt.figure()
            sequence_len = np.squeeze(original_seq).shape[0]
            xaxis = np.arange(1,seq_len+1)/cfg.dataset.fs 
            waves = np.zeros((sequence_len,5))
            fmm_dict = convert_fmm_array_to_dict(fmm_array=fmm_coeff_array,num_leads=cfg.dataset.num_features,num_waves=5)
            for i,wave_name in enumerate(["P","Q","R","S","T"]):
                wave = np.squeeze(generate_wave(fmm_dict,wave_name=wave_name,lead=lead,seq_len=seq_len))
                waves[0:seq_len,i]=wave
            plt.plot(xaxis,original_seq[:seq_len,lead],label=f"original",color="b",linewidth=2.0)
            plt.plot(xaxis,fmm_dict["P"]["M"][lead] + np.sum(waves,axis=1)[:seq_len],label=f"Reconstruction",color="r", linewidth=2.0)
            for j,w in enumerate(["P","Q","R","S","T"]):
                plt.plot(xaxis,np.squeeze(waves[:,j])[:seq_len],linestyle="dashed", linewidth=1.0)
            plt.xlabel("Time [s]")
            plt.ylabel("ECG")
            # plt.title(f"Data sample index: {sample_index}, class: {seq_label}")
            plt.legend(loc="best",fontsize=9)
            save_png_eps_figure(f"fmm_waves_original_{sample_index}_class_{seq_label}_lead_{lead}")

    # %%
    # Preprocess dataset
    train_dict = call(cfg.dataset.preprocess_function, input_data=raw_dataset_dict["train"], dataset_params=raw_dataset_dict["params"]) 
    test_dict = call(cfg.dataset.preprocess_function, input_data=raw_dataset_dict["test"], dataset_params=raw_dataset_dict["params"]) 
    if(cfg.model.name in non_ae_baseline_models):
        # Get train and validation dataset
        train_dataset, val_dataset = get_only_normal_train_val_dataset(in_dict=train_dict, normal_class=normal_class,
                                                                        batch_size=cfg.batch_size, seed=cfg.seed, 
                                                                        only_normal=cfg.dataset.select_only_normal, val_size=0.0,
                                                                        return_type="dict")
        test_dataset = test_dict
    else:
        # Get train and validation dataset
        train_dataset, val_dataset = get_only_normal_train_val_dataset(in_dict=train_dict, normal_class=normal_class,
                                                                        batch_size=cfg.batch_size, seed=cfg.seed, 
                                                                        only_normal=cfg.dataset.select_only_normal, val_size=cfg.dataset.val_size)
        #Get test dataset
        test_dataset = Dataset.from_tensor_slices(test_dict).batch(cfg.batch_size,drop_remainder=True).prefetch(10)
    # Compute coefficients parameters for the dataset and set them in the configuration file
    if(cfg.model.name in fmm_model_names):
        if(cfg.dataset.name in ["ptb_xl_fmm","shaoxing_fmm"]):
            train_coeffs = train_dict["coefficients"][train_dict["labels"]==normal_class,:] if cfg.dataset.select_only_normal else train_dict["coefficients"]
            cfg.model.coeffs_properties_dict = get_statistics_dict_from_matrix(train_coeffs)
        elif(cfg.dataset.name in ["ecg5000"]):
            cfg.model.coeffs_properties_dict = None


    # %%
    # Test if model inference works
    model = instantiate(cfg.model)
    model.test_step({k: v[:cfg.batch_size] for k,v in train_dict.items()})

    # %%
    # Create and compile model
    model = instantiate(cfg.model)
    optimizer = instantiate(cfg.optimizer)
    model.compile(optimizer=optimizer)
    # Learning rate scheduler with decay
    def scheduler(epoch, lr):
        if epoch < 5:
            return lr
        elif epoch < 15:
            return lr * tf.math.exp(-0.05)
        else:
            return lr * tf.math.exp(-0.001)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history_warmup = None
    warmup_time = 0.0   # Needed to keep track of training times
    warmup_epoch_training_times = []
    if(model.need_warmup):
        #Warmup phase for coefficient regression
        print("Starting warmup")
        lr_callback_warmup = tf.keras.callbacks.LearningRateScheduler(scheduler)
        callbacks_warmup = [
        instantiate(cfg.dataset.es_callback),
        lr_callback_warmup,
        cp_cb_generator(os.path.join(cfg.tb_output_dir,"checkpoint_warmup")),
        tf.keras.callbacks.TensorBoard(log_dir=cfg.tb_output_dir, histogram_freq=1),
        TrainingTimeCallback()
        ] 
        if(cfg.model.name in ["cvae"]):
            #Only reconstruction error phase as warmup for variational autoencoder (no KL divergence loss)
            model.alpha, model.beta = 1.0, 0.0
        elif(cfg.model.name in fmm_model_names):
            model.reconstruction_loss_weight, model.coefficient_loss_weight = 0.0, 1.0
        start_warmup_time = time.time()
        history_warmup = model.fit(train_dataset,epochs=cfg.model.num_warmup_epochs,
                                validation_data=val_dataset,callbacks=callbacks_warmup)
        end_warmup_time = time.time()
        warmup_time = end_warmup_time-start_warmup_time
        warmup_epoch_training_times = callbacks_warmup[4].epoch_times
        model = instantiate(cfg.model)
        model.load_weights(os.path.join(cfg.tb_output_dir,"checkpoint_warmup"))
        optimizer = instantiate(cfg.optimizer)
        model.compile(optimizer=optimizer)
        K.set_value(model.optimizer.lr, cfg.optimizer.learning_rate)
        if(cfg.model.name in ["cvae"]):
            model.alpha, model.beta = 1.0, 1.0
        elif(cfg.model.name in fmm_model_names):
            model.reconstruction_loss_weight, model.coefficient_loss_weight = 1.0, 0.0
        print("Ending warmup")
    callbacks = [
        instantiate(cfg.dataset.es_callback),
        lr_callback,
        cp_cb_generator(os.path.join(cfg.tb_output_dir,"checkpoint_ad")),
        tf.keras.callbacks.TensorBoard(log_dir=cfg.tb_output_dir, histogram_freq=1),
        TrainingTimeCallback()
    ]
    train_time = 0.0
    train_epoch_training_times = []
    history = None
    if(cfg.train.num_epochs>0):
        start_train_time = time.time()
        history = model.fit(train_dataset,epochs=cfg.train.num_epochs,validation_data=val_dataset,callbacks=callbacks)
        end_train_time = time.time()
        train_time = end_train_time-start_train_time
        train_epoch_training_times = callbacks[4].epoch_times
    save_dict({"train_time":train_time,"warmup_time":warmup_time,
            "warmup_epochs_time":warmup_epoch_training_times,
            "train_epochs_time":train_epoch_training_times,
            },"train_time")

    # %%
    # Load best model from checkpoint
    # cfg.model.coeffs_properties_dict = None
    model = instantiate(cfg.model)
    try:
        checkpoint_path = os.path.join(cfg.tb_output_dir,"checkpoint_ad") #Restore best model 
        model.load_weights(checkpoint_path).expect_partial()
    except:
        print("Loading warmup model because no anomaly detection checkpoint was found")
        checkpoint_path = os.path.join(cfg.tb_output_dir,"checkpoint_warmup") #Restore best model in only warmup case
        model.load_weights(checkpoint_path).expect_partial()

    # %%
    if(cfg.model.name in fmm_model_names):
        loss_matrix_dict = get_loss_from_fmm_model(model,test_dataset,cfg,True)
        print(np.mean(loss_matrix_dict["loss_matrix"],axis=0))

    # %%
    # Plot waves and get predicted coefficient for TEST dataset
    if(cfg.model.name not in non_ae_baseline_models):
        predict_results_dict = model.predict(test_dataset)
        original = predict_results_dict["data"]
        reconstruction = predict_results_dict["predicted_data"]
        sample_index_list = np.random.randint(0,len(test_dict["inputs"]),(20))
        lead = 0 # Plot always the first lead if there are multiple
        if(cfg.model.name in fmm_model_names):
            waves,fmm_coeff_matrix = get_waves_from_fmm_model(model,test_dataset,cfg)
            m_start_index,m_end_index = get_M_indexes(wave_index=None,num_leads=cfg.dataset.num_features,num_waves=5)
            for sample_index in sample_index_list:
                plt.figure()
                sample_zero_indexes = np.where(original[sample_index,:,lead]==0)[0]
                sequence_len = sample_zero_indexes[0] if (len(sample_zero_indexes)>0) else original.shape[1]
                xaxis = np.arange(1,sequence_len+1)/cfg.dataset.fs
                sample_label = leg[test_dict["labels"][sample_index]]
                plt.plot(xaxis,np.squeeze(original[sample_index,:sequence_len,lead]),label=f"original",color="b",linewidth=2.0)
                plt.plot(xaxis,fmm_coeff_matrix[sample_index,m_start_index+lead] + np.sum(waves[sample_index,:sequence_len,lead,:],axis=1),label=f"reconstruction",color="r", linewidth=2.0)
                for j,w in enumerate(["P","Q","R","S","T"]):
                    plt.plot(xaxis,np.squeeze(waves[sample_index,:sequence_len,lead,j]),linestyle="dashed", linewidth=1.0)
                plt.xlabel("Time [s]")
                plt.ylabel("ECG")
                plt.title(f"Data sample index: {sample_index}, class: {sample_label}, lead: {lead}")
                plt.legend(loc="upper right")
                save_png_eps_figure(f"fmm_waves_{sample_index}_class_{sample_label}_lead_{lead}")

    # %%
    # Plot parameter histograms for each lead
    if(cfg.model.name in fmm_model_names and cfg.dataset.name in ["ptb_xl_fmm","shaoxing_fmm"]):
        beta_indexes = [get_beta_indexes(wave_index=i,num_leads=cfg.dataset.num_features,num_waves=cfg.dataset.num_waves)[0] for i in range(cfg.dataset.num_waves)]
        m_index = get_M_indexes(wave_index=i,num_leads=cfg.dataset.num_features,num_waves=cfg.dataset.num_waves)[0]
        a_indexes = [get_A_indexes(wave_index=i,num_leads=cfg.dataset.num_features,num_waves=cfg.dataset.num_waves)[0] for i in range(cfg.dataset.num_waves)]
        alpha_indexes = [get_alpha_indexes(wave_index=i,num_leads=cfg.dataset.num_features,num_waves=cfg.dataset.num_waves)[0] for i in range(cfg.dataset.num_waves)]
        omega_indexes = [get_omega_indexes(wave_index=i,num_leads=cfg.dataset.num_features,num_waves=cfg.dataset.num_waves)[0] for i in range(cfg.dataset.num_waves)]
        for fmm_model_coefficients,fmm_model_type in zip([test_dict["coefficients"],fmm_coeff_matrix],["Original", "Predicted"]):
            for f,coeff_name in zip([get_A_indexes,get_alpha_indexes,get_beta_indexes,get_omega_indexes],["A","Alpha","Beta","Omega"]):
                for wave_index in range(cfg.dataset.num_waves):
                    num_lead_indexes_per_parameter = 1 if coeff_name in ["Alpha", "Omega"] else cfg.dataset.num_features
                    for lead_index in range(num_lead_indexes_per_parameter):
                        coeff_index = f(wave_index=wave_index,num_leads=cfg.dataset.num_features,num_waves=cfg.dataset.num_waves)[0]+lead_index
                        plt.figure()
                        vals = fmm_model_coefficients[test_dict["labels"]==normal_class,coeff_index]
                        plt.hist(vals,bins=50)
                        plt.title(f"{fmm_model_type} {coeff_name}_{wave_index}_{lead_index}")
                        plt.xlabel("Value")
                        plt.ylabel("Occurrences")
                        print(coeff_name,wave_index,np.average(vals),np.std(vals),np.average(vals)+np.std(vals))
                        save_png_eps_figure(f"fmm_histogram_coeff_{fmm_model_type}_{coeff_name}_wave_{wave_index}_lead_{lead_index}")
            for lead_index in range(cfg.dataset.num_features):
                coeff_index = get_M_indexes(wave_index=None,num_leads=cfg.dataset.num_features,num_waves=cfg.dataset.num_waves)[0] + lead_index
                plt.figure()
                vals = fmm_model_coefficients[test_dict["labels"]==normal_class,coeff_index]
                plt.hist(vals,bins=50)
                plt.title(f"{fmm_model_type} M, lead: {lead_index}")
                plt.xlabel("Value")
                plt.ylabel("Occurrences")
                save_png_eps_figure(f"fmm_histogram_coeff_{fmm_model_type}_M_wave_{wave_index}")

    # %%
    # Plot correlation of predicted coefficients to the original coefficients (obtained through FMM optimization)
    if(cfg.model.name in fmm_model_names and cfg.dataset.name in ["ptb_xl_fmm","shaoxing_fmm"]):
        original_test_fmm_coefficients = test_dict["coefficients"]
        sorted_test_predicted_coefficients = sort_fmm_coeffs_array(fmm_coeff_matrix, num_leads=cfg.dataset.num_features, num_waves=5)
        sorted_test_coefficients = sort_fmm_coeffs_array(original_test_fmm_coefficients, num_leads=cfg.dataset.num_features, num_waves=5)
        coefficients_names = get_parameters_names_list(cfg.dataset.num_features,cfg.dataset.num_waves)
        circular_indexes = get_circular_indexes_as_boolean_t(cfg.dataset.num_features,cfg.dataset.num_waves).numpy()
        correlation_coefficients_pred_orig_fmm_no_sort = mixed_lin_circ_corr_coeff(
                    x=original_test_fmm_coefficients[test_dict["labels"]==normal_class], \
                    y=fmm_coeff_matrix[test_dict["labels"]==normal_class],
                    c=circular_indexes)
        correlation_coefficients_pred_orig_fmm_no_sort_dict = {n:v for n,v in zip(coefficients_names,correlation_coefficients_pred_orig_fmm_no_sort)}
        filename = os.path.join(cfg.tb_output_dir,f"correlation_normal_no_sorted")
        json.dump(correlation_coefficients_pred_orig_fmm_no_sort_dict, open(filename,"w"),indent=0) 
        df = pd.DataFrame([correlation_coefficients_pred_orig_fmm_no_sort_dict])
        print(correlation_coefficients_pred_orig_fmm_no_sort_dict)
        sorted_test_coefficients_normal = sorted_test_coefficients[test_dict["labels"]==normal_class]
        sorted_test_predicted_coefficients_normal = sorted_test_predicted_coefficients[test_dict["labels"]==normal_class]
        correlation_coefficients_pred_orig_fmm = mixed_lin_circ_corr_coeff(
                    x=sorted_test_coefficients_normal, \
                    y=sorted_test_predicted_coefficients_normal,
                    c=circular_indexes)
        correlation_coefficients_pred_orig_fmm_dict = {n:v for n,v in zip(coefficients_names,correlation_coefficients_pred_orig_fmm)}
        filename = os.path.join(cfg.tb_output_dir,f"correlation_normal_sorted")
        json.dump(correlation_coefficients_pred_orig_fmm_no_sort_dict, open(filename,"w"),indent=0) 
        print(list(zip(coefficients_names,correlation_coefficients_pred_orig_fmm)))
        #Plot dictionaries and save them
        plot_scalar_dictionary(correlation_coefficients_pred_orig_fmm_no_sort_dict)
        save_png_eps_figure("pred_orig_corr_coeff_no_sorted")
        plot_scalar_dictionary(correlation_coefficients_pred_orig_fmm_dict)
        save_png_eps_figure("pred_orig_corr_coeff_sorted")

    # %%
    if(cfg.model.name in fmm_model_names and cfg.dataset.name in ["ptb_xl_fmm","shaoxing_fmm"]):
        for sample_index in sample_index_list:
            to_plot_original = sorted_test_coefficients[sample_index]
            to_plot_label = leg[test_dict["labels"][sample_index]]
            plt.figure()
            sample_len = int(test_dict["sizes"][sample_index])
            xaxis = np.arange(1,sample_len+1)/cfg.dataset.fs 
            plt.plot(xaxis,test_dict["inputs"][sample_index, :sample_len, lead], label="ECG Input")
            plot_fmm_wave_from_coefficients(fmm_coeff_array=to_plot_original,
                                            num_leads=cfg.dataset.num_features,
                                            seq_len=sample_len,
                                            fs=cfg.dataset.fs,
                                            lead=lead,
                                            add_single_waves=False,
                                            label="Original FMM")
            plt.title("Original sorted FMM coefficients")
            sorted_test_predicted_coefficients = sort_fmm_coeffs_array(fmm_coeff_matrix, num_leads=cfg.dataset.num_features, num_waves=5)
            to_plot_pred = sorted_test_predicted_coefficients[sample_index]
            plot_fmm_wave_from_coefficients(fmm_coeff_array=to_plot_pred,
                                            num_leads=cfg.dataset.num_features,
                                            seq_len=sample_len,
                                            fs = cfg.dataset.fs,
                                            lead=lead,
                                            add_single_waves=False,
                                            label="Predicted FMM")
            plt.title(f"Predicted sorted FMM coefficients, sample {sample_index}, class {to_plot_label}, lead: {lead}")
        sample_index=100
        coefficients_dict = {}
        for a,b,name in zip(sorted_test_coefficients_normal[sample_index],
                            sorted_test_predicted_coefficients_normal[sample_index],
                            coefficients_names):
            print(f"{name}: \n \t original: {a}, \n \t predicted: {b}")

    # %%
    if(cfg.model.name not in non_ae_baseline_models):
        for sample_index in sample_index_list:
            plt.figure()
            sequence_len = np.squeeze(original[sample_index,:,lead]).shape[0]
            sample_len = int(test_dict["sizes"][sample_index])
            xaxis = np.arange(1,sample_len+1)/cfg.dataset.fs
            sample_label = leg[test_dict["labels"][sample_index]]
            plt.plot(xaxis,np.squeeze(original[sample_index,:sample_len,lead]),label=f"{sample_index}",color="b")
            plt.xlabel("Time [s]")
            plt.ylabel("ECG")
            plt.title(f"Data sample index: {sample_index}, class: {sample_label}")
            if(cfg.save_plots):
                filename = os.path.join(cfg.tb_output_dir,f"original_{sample_index}")
                plt.savefig(filename+".png")
                plt.savefig(filename+".eps")
            plt.figure()
            plt.plot(xaxis,np.squeeze(reconstruction[sample_index,:sample_len,lead]),label=f"{sample_index}",color="r")
            plt.xlabel("Time [s]")
            plt.ylabel("ECG")
            plt.title(f"Data sample index: {sample_index}, class: {sample_label}")
            save_png_eps_figure(f"reconstruction_{sample_index}")

    # %%
    if(cfg.model.name not in non_ae_baseline_models):
        for sample_index in sample_index_list:
            plt.figure()
            sequence_len = np.squeeze(original[sample_index,:,lead]).shape[0]
            xaxis = np.arange(1,sequence_len+1)/cfg.dataset.fs
            sample_label = leg[test_dict["labels"][sample_index]]
            plt.plot(xaxis,np.squeeze(original[sample_index,:,lead]),label=f"original_{sample_index}",color="b")
            plt.plot(xaxis,np.squeeze(reconstruction[sample_index,:,lead]),label=f"reconstruction_{sample_index}",color="r")
            plt.xlabel("Time [s]")
            plt.ylabel("ECG")
            plt.title(f"Data sample index: {sample_index}, class: {sample_label}, lead {lead}")
            plt.legend()
            if(cfg.save_plots):
                filename = os.path.join(cfg.tb_output_dir,f"original_recons_{sample_index}")
                plt.savefig(filename+".png")
                plt.savefig(filename+".eps")

    # %%
    if(cfg.model.name not in non_ae_baseline_models):
        plt.figure()
        for sample_index in range(10):
            sequence_len = np.squeeze(original[sample_index]).shape[0]
            xaxis = np.arange(1,sequence_len+1)/cfg.dataset.fs
            plt.plot(xaxis,np.squeeze(original[sample_index,:,lead]),label=f"{sample_index}",color="b")
            plt.plot(xaxis,np.squeeze(reconstruction[sample_index,:,lead]),label=f"{sample_index}",color="y")
        plt.title("Model prediction vs original data")
        plt.xlabel("Time [s]")
        plt.ylabel("ECG")
        plt.legend()
        save_png_eps_figure(f"reconstruction_check")

    # %%
    # Plot training history
    def history_plot_fun(history, file_name):
        plt.figure()
        plt.rcParams.update({'font.size': 8})
        if(cfg.model.name=="rythm_vae"):
            plot_list = [['loss','val_loss'],
                        ['morpho_reconstruction_loss','rythm_reconstruction_loss','val_morpho_reconstruction_loss','val_rythm_reconstruction_loss'],
                        ['morpho_kl_loss','rythm_kl_loss','val_morpho_kl_loss','val_rythm_kl_loss']]
        elif(cfg.model.name in["bert_ecg", "conv_ae", "encdec_ad", "dense_ae", "lstm_ae", "ecgnet"] ):
            plot_list = [['loss','val_loss']]
        elif(cfg.model.name in fmm_model_names):
            plot_list = [['loss','val_loss'],['reconstruction_loss','val_reconstruction_loss'],['coefficient_loss','val_coefficient_loss']]
        elif(cfg.model.name=="ecg_adgan"):
            plot_list = [['D_loss','G_loss'],['acc']]
        elif(cfg.model.name=="diffusion_ae"):
            plot_list = [['sum_loss_train','ae_loss_train','diff_loss_train']]
        else:
            plot_list = [['loss','val_loss'],['reconstruction_loss','val_reconstruction_loss'],['kl_loss','val_kl_loss']]
        for i,metric_to_plot_list in enumerate(plot_list):
            ax = plt.subplot(len(plot_list),1,i+1)
            for metric_to_plot in metric_to_plot_list:
                to_plot = history.history[metric_to_plot]
                ax.plot(to_plot,label=metric_to_plot)
            plt.legend()
            ax.set_title(metric_to_plot_list[0].capitalize())
            ax.set_ylabel("Loss")
            if(i!=len(plot_list)-1):
                ax.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False)
        ax.set_xlabel("Epoch")
        if(cfg.save_plots):
            filename = os.path.join(cfg.tb_output_dir,f"{file_name}.json")
            with open(filename,"wb") as f: 
                pickle.dump(history.history,f)
            save_png_eps_figure(f"{file_name}")
    if(history is not None):
        history_plot_fun(history=history, file_name="history")
    if(history_warmup is not None):
        history_plot_fun(history=history_warmup, file_name="history_warmup")

    # %%
    # Compute train and test losses
    if(cfg.model.name not in non_ae_baseline_models):
        train_class_loss_dict = get_loss_score(in_vae=model,data_dict=train_dict,batch_size=cfg.batch_size,classes=leg,num_samples=-1)
        test_class_loss_dict = get_loss_score(in_vae=model,data_dict=test_dict,batch_size=cfg.batch_size,classes=leg,num_samples=-1)
    else:
        train_class_loss_dict = model.compute_class_loss(train_dict, leg)
        test_class_loss_dict = model.compute_class_loss(test_dict, leg)
    print(f"Train: {train_class_loss_dict} \nTest:{test_class_loss_dict}")
    print("Normal class mean accuracy train/test : {0},{1}".format(train_class_loss_dict["mean"][normal_class], test_class_loss_dict["mean"][normal_class]))
    if(cfg.save_plots):
        filename = os.path.join(cfg.tb_output_dir,f"train_class_loss")
        json.dump(train_class_loss_dict, open(filename,"w"),indent=0)  
        filename = os.path.join(cfg.tb_output_dir,f"test_class_loss")
        json.dump(test_class_loss_dict, open(filename,"w"),indent=0)   

    # %%
    # Compute confusion matrix for an example threshold
    threshold = train_class_loss_dict["mean"][normal_class]+0.05*train_class_loss_dict["std"][normal_class]
    if(cfg.model.name not in non_ae_baseline_models):
        test_confusion_matrix = get_confusion_matrix(in_vae=model,data_dict=test_dict,
                                                    batch_size=cfg.batch_size,threshold=threshold,
                                                    classes=leg,normal_class=normal_class)
    else:
        test_confusion_matrix = model.get_confusion_matrix(test_dict, threshold, normal_class)
    print(test_confusion_matrix)


    # %%
    # Compute train and test roc curve
    train_roc_filename = os.path.join(cfg.tb_output_dir,f"train_roc") if cfg.save_plots else None 
    test_roc_filename = os.path.join(cfg.tb_output_dir,f"test_roc") if cfg.save_plots else None
    if(cfg.model.name not in non_ae_baseline_models):
        train_roc_dict = get_roc_auroc(model, train_dict, cfg.batch_size, normal_class, filename=train_roc_filename)
        test_roc_dict = get_roc_auroc(model, test_dict, cfg.batch_size, normal_class, filename=test_roc_filename)
    else:
        train_roc_dict = model.compute_roc(train_dict, normal_class, train_roc_filename)
        test_roc_dict = model.compute_roc(test_dict, normal_class, test_roc_filename)
    train_auc,test_auc = train_roc_dict["roc_auc"], test_roc_dict["roc_auc"]
    print(f"Train AUC: {train_auc} \nTest AUC: {test_auc}")
    if(cfg.save_plots):
        filename = os.path.join(cfg.tb_output_dir,f"train_roc.json")
        json.dump(train_roc_dict, open(filename,"w"),indent=0)  
        filename = os.path.join(cfg.tb_output_dir,f"test_roc.json")
        json.dump(test_roc_dict, open(filename,"w"),indent=0)  

    # %%
    # Plot some original and reconstructed ECGs
    if(cfg.model.name not in non_ae_baseline_models):
        filename = os.path.join(cfg.tb_output_dir,"orig_plus_rec_train_sample") if cfg.save_plots else None
        plot_one_ecg_for_class(in_vae=model,data_dict=train_dict,classes=leg,filename=filename,fs=cfg.dataset.fs,batch_size=cfg.batch_size, lead=lead)
        filename = os.path.join(cfg.tb_output_dir,"orig_plus_rec_test_sample") if cfg.save_plots else None
        plot_one_ecg_for_class(in_vae=model,data_dict=test_dict,classes=leg,filename=filename,fs=cfg.dataset.fs,batch_size=cfg.batch_size, lead=lead)

    # %%
    # Compute number of parameters and size of the model
    if(cfg.model.name not in non_ae_baseline_models or cfg.model.name in "ecg_adgan"):
        trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights],dtype=int)
        non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights],dtype=int)
    else:
        trainable_count = model.get_number_trainable_parameters()
        non_trainable_count = model.get_number_non_trainable_parameters()
    total_num_params = trainable_count + non_trainable_count
    # Get the size of the model file on disk
    if(cfg.model.name not in non_ae_baseline_models):
        model_size_bytes = os.path.getsize(checkpoint_path+".data-00000-of-00001")
    else:
        model_size_bytes = model.get_model_size(os.path.dirname(callbacks[2].filepath))
    model_size_mb = model_size_bytes / (1024 * 1024)
    model_size_dict = {"num_trainable": trainable_count,
                    "num_non_trainable": non_trainable_count,
                        "num_parameters": total_num_params,
                        "model_size_bytes": model_size_bytes,
                        "model_size_mb": model_size_mb,
                        }
    save_dict(model_size_dict,"model_size")
    print(model_size_dict)
    # load_dict_from_binary(os.path.join(cfg.tb_output_dir,"model_size"))
    # Compute inference times
    inference_batch_size = 16
    if(cfg.model.name not in non_ae_baseline_models):
        dataset = Dataset.from_tensor_slices(test_dict).batch(inference_batch_size) #Use always batch size 16 to compute inference times
    else:
        model.batch_size = 16
        dataset = test_dataset
    num_samples = len(dataset)  # Number of input samples
    start_predict_time = time.time()
    model.predict(dataset, verbose=None)
    end_predict_time = time.time()
    predict_time = end_predict_time - start_predict_time
    save_dict({"num_batches":num_samples,"predict_time":predict_time,
            "avg_predict_time":predict_time/num_samples,"batch_size" :inference_batch_size},"predict_time")
    print(predict_time)
    # load_dict_from_binary(os.path.join(cfg.tb_output_dir,"predict_time"))
    # inference_times = []
    # for in_data in dataset:
    #     start_time = time.time()  # Record start time
    #     predictions = model.predict(in_data, verbose=None)  # Perform inference
    #     end_time = time.time()  # Record end time
    #     inference_time = end_time - start_time
    #     inference_times.append(inference_time)
    # inference_times = np.array(inference_times)
    # save_np(inference_times, "inference_times")
    # np.load(os.path.join(cfg.tb_output_dir,"inference_times.npy"))

    # %%
    # Set completed in configuration file and save it in the experiment folder
    cfg.completed=True
    filename = os.path.join(cfg.tb_output_dir,f"conf.yaml")
    with open(filename,"w") as fp:
        OmegaConf.save(config=cfg, f=fp)

if __name__ == '__main__':
    silence_tensorflow()
    my_app()
