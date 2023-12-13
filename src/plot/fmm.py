import numpy as np
from src.utils.fmm import convert_fmm_array_to_dict, generate_wave
import matplotlib.pyplot as plt

def plot_fmm_wave_from_dict(fmm_coeff_dict:np.ndarray, seq_len:int, fs:int, lead:int , seq_label=None, add_single_waves:bool=True,label="fmm 3d model")->None:
        # assert lead==0 # Only one lead is supported right now
        xaxis = np.arange(1,seq_len+1)/fs 
        waves = np.zeros((seq_len,5))
        # fmm_dict = convert_fmm_array_to_dict(fmm_array=fmm_coeff_array,num_leads=1,num_waves=5)
        for i,wave_name in enumerate(["P","Q","R","S","T"]):
            wave = np.squeeze(generate_wave(fmm_coeff_dict,wave_name=wave_name,lead=lead,seq_len=seq_len))
            waves[0:seq_len,i]=wave
        # plt.plot(xaxis,np.squeeze(original_seq),label=f"original",color="b",linewidth=2.0)
        plt.plot(xaxis,fmm_coeff_dict["P"]["M"][lead] + np.sum(waves,axis=1),label=label, linewidth=2.0)
        if(add_single_waves):
            for j,w in enumerate(["P","Q","R","S","T"]):
                plt.plot(xaxis,np.squeeze(waves[:,j]),label=f"wave {w}",linestyle="dashed", linewidth=1.0)
        plt.legend()

def plot_fmm_wave_from_coefficients(fmm_coeff_array:np.ndarray, 
                                    seq_len:int, 
                                    num_leads, 
                                    fs:int, 
                                    lead: int, 
                                    seq_label=None, 
                                    add_single_waves:bool=True,
                                    label:str="fmm 3d model")->None:
    fmm_coeff_dict = convert_fmm_array_to_dict(fmm_array=fmm_coeff_array,num_leads=num_leads,num_waves=5)
    plot_fmm_wave_from_dict(fmm_coeff_dict=fmm_coeff_dict,
                            seq_len=seq_len, 
                            fs=fs, 
                            lead=lead, 
                            seq_label=seq_label, 
                            add_single_waves=add_single_waves,
                            label=label)

if __name__ == '__main__':
    pass