#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dataset> <model> <num_experiments>"
    exit 1
fi

# Assign input arguments to variables
dataset="$1"
model="$2"
num_experiments="$3"

# Set options for specific datasets
if [ "$dataset" = "ptb_xl_fmm" ]; then
    dataset_options="dataset.load_function.lead=1 dataset.num_features=1 dataset.sequence_length=300 batch_size=64"
elif [ "$dataset" = "shaoxing_fmm" ]; then
    dataset_options="dataset.load_function.lead=1 dataset.num_features=1 dataset.sequence_length=1000"
else
    dataset_options=""
fi

# Set batch size option based on the model
if [ "$model" = "ecgnet" ] || [ "$model" = "lstm_ae" ] || \
   [ "$model" = "encdec_ad" ] || [ "$model" = "fmm_encdec_ad" ] || \
   [ "$model" = "fmm_ecgnet" ] || [ "$model" = "fmm_lstm_ae" ]; then
    batch_size_option="batch_size=128"
elif [ "$model" = "fmm_cae" ] || [ "$model" = "cvae" ]; then
    batch_size_option="batch_size=64"
else
    batch_size_option=""
fi

# Set batch size for specific dataset
if [ "$dataset" = "ecg5000" ]; then
    batch_size_option="batch_size=16"
    if [ "$model" = "fmm_encdec_ad" ] || [ "$model" = "fmm_cae" ] || \
    [ "$model" = "fmm_dense_ae" ] || [ "$model" = "fmm_bert_ecg" ] || \
    [ "$model" = "fmm_lstm_ae" ]; then
        batch_size_option="batch_size=16 dataset.split_ecg=True model.num_warmup_epochs=0"
    fi
fi

# ECG-ADGAN and DiffusionAE have different parameters that override the previous ones 
if [ "$model" = "ecg_adgan" ]; then
    batch_size_option="batch_size=128 optimizer=legacy_adam train.num_epochs=20000"
elif [ "$model" = "diffusion_ae" ]; then
    batch_size_option="batch_size=32 optimizer=adam_torch train.num_epochs=500"
fi

# List of learning rates to iterate through
learning_rates=("0.001" "0.0005" "0.0001" "0.00005" "0.00001") 

# Define an array of seeds
seeds=("123" "456" "789" "101112" "131415" "161718" "192021")

# Activate Conda environment
conda activate sfl

# Get total number of available GPUs
num_gpus=$(nvidia-smi -L | wc -l)

# Loop through each learning rate
for lr in "${learning_rates[@]}"; do
    echo "Training with learning rate $lr"
    
    # Loop through each experiment and use corresponding seed
    for ((i = 0; i < num_experiments; i++)); do
        echo "Experiment $((i+1))"
        seed="${seeds[i]}"
        
        # Find the next available GPU index (cycle if needed)
        gpu_idx=$((i % num_gpus))
        
        # Launch the script using the selected GPU
        CUDA_VISIBLE_DEVICES=$gpu_idx python ecg_anomaly_detection.py dataset=$dataset model=$model optimizer.learning_rate=$lr seed=$seed $dataset_options $batch_size_option &
        
        # Introduce a 10-second delay before launching the next experiment
        sleep 10
    done
    
    # Wait for all scripts launched for this learning rate to finish before proceeding
    wait
done

# Deactivate Conda environment
# conda deactivate
