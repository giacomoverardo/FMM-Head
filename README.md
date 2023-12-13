<!-- # ScalableFederatedLearning
Code for scalable federated learning project funded by VR -->

## FMM-Head

This repository contains the code for the "FMM-Head: Enhancing Autoencoder-based ECG anomaly detection with prior knowledge" paper [2]. Please [cite our paper](#citation) if you use our code.
### Setup environment

1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Create a conda environment. Use the following command to create a new environment named "fmmhead" (replace "fmmhead" with your desired environment name):

        conda env create -f ./env-files/fmm-head-environment.yml --name fmmhead 

We provide 3 equivalent ways to run fmm-head: 
1. A python script [ecg_anomaly_detection.py](ecg_anomaly_detection.py) 
2. A notebook [ecg_anomaly_detection.ipynb](ecg_anomaly_detection.ipynb) useful for interactivity
3. Bash scripts [run_experiments.sh](run_experiments.sh) and [run_experiments_parallel.sh](run_experiments_parallel.sh) to launch many simulations with available datasets, models and learning rates at the same time.

### Running in Python
Run [ecg_anomaly_detection.py](/ecg_anomaly_detection.py) in a terminal with datasets and models from [the following section](#configurations) and the arguments you can find in [the configuration file](/conf/ecg_anomaly_detection.yaml). For instance, to run the model *fmm_dense_ae* on the *ecg5000* dataset, learning rate 0.00003, 500 epochs, batch size 32:

        conda activate fmmhead
        python ecg_anomaly_detection.py dataset=ecg5000 model=fmm_dense_ae batch_size=32 train.num_epochs=500 optimizer.learning_rate=0.00003


### Running in notebook
The main notebook is [ecg_anomaly_detection.ipynb](/ecg_anomaly_detection.ipynb). You can update the configuration by modifying the overrides arguments for the hydra `compose` function in the first cell:

        cfg=compose(config_name='ecg_anomaly_detection.yaml',return_hydra_config=True,
                overrides=["hydra.verbose=true","dataset=ecg5000",
                                "model=fmm_dense_ae","batch_size=32",
                                "optimizer=adam",
                                "optimizer.learning_rate=0.00003", 
                                "train.num_epochs=500",
                                "save_plots=True])  

### Running multiple FMM-Head experiments
In a terminal run the following [bash script](/run_experiments.sh):

        ./run_experiments.sh <dataset> <model> <num_experiments> <gpu_index> [lr1] [lr2] ..."

If no learning rates are specified, the predefined range from 0.001 to 0.00001 will be used. If you have more than 1 GPU and want to launch experiments in parallel, use instead the [ parallelized-version](/run_experiments_parallel.sh):

        ./run_experiments_parallel <dataset> <model> <num_experiments>

### Available models, datasets and additional configurations
We use [hydra](https://hydra.cc/docs/intro/) to handle configuration file. Go to [the configuration file](/conf/ecg_anomaly_detection.yaml) and modify the ??? fields with your own paths. 

The following datasets are available (see [the FMM-Head paper](#citation) for additional information): 

- *shaoxing_fmm*: the Shaoxing dataset with correspondent FMM parameters. 
- *ptb_xl_fmm*: the PTB-XL dataset with correspondent FMM parameters. 
- *ecg5000*: a standard 1-patient, 1-lead dataset.

Note: *shaoxing_fmm* and *ptb_xl_fmm* are large datasets, so you will need around 1GB of available space for each of them on your dataset. 
The following baselines are available (their description and citation is provided in [the FMM-Head paper](#citation)): 

- *bert_ecg*: a transformer based autoencoder
- *cvae*: a convolutional variational autoencoder
- *encdec_ad*: an lstm based autoencoder
- *ecgnet*: an lstm based autoencoder
- *dense_ae*: an fully connected based autoencoder
- *ecg_adgan*: a GAN model for ECG anomaly detection
- *diffusion_ae*: a diffusion model for time series anomaly detection. 

To run the correspondent FMM model (only for the 5 autoencoders), use the prefix *fmm_*. For instance *fmm_dense_ae*.

For additional parameters, please refer to the [configuration file](/conf/ecg_anomaly_detection.yaml).

### Running DiffusionAE
DiffusionAE is implemented in torch and requires a different environment:

        conda env create -f ./env-files/diffusion-ae-environment.yml --name diffusionae

Use the diffusionae environment instead of fmmhead when you provide *diffusion_ae* as argument for the previous scripts.


### Collecting results
The output data, images and models are saved into the `tb_output_dir` path of the [configuration file](/conf/ecg_anomaly_detection.yaml). You can collect results from multiple runs by means of functions similairs to the ones provided in the [collect_results notebook](/collect_results.ipynb).

### Generate FMM dataset (Optional)
When running the previous code for the PTB-XL and Shaoxing datasets you automatically download the heartbeat segmented, preprocessed version with the extracted FMM coefficients. **Extracting the FMM coefficients takes days/weeks depending on your computational capabilities**. To do it yourself you will need to run the code for the extraction of the FMM paramateres:

1. Install the [R software](https://www.r-project.org/) either from the website or in conda
2. Clone the [FMMECG3D repository](https://github.com/FMMGroupVa/FMMECG3D/tree/main) [1]  by running:

        git clone git@github.com:FMMGroupVa/FMMECG3D.git <your_custom_fmm_r_repository_path>
3. Add in the [dataset generation configuration file](conf/generate_fmm_ds.yaml) the FMMECG3D path, R path and data path.
4. Add the same information in the [dataset generation python script](generate_fmm_ds.py)
5. Run the script on Shaoxing or PTB-XL in your conda environment:

        python generate_fmm_ds.py dataset=shaoxing
<!-- ### Prerequisites 

1. Install conda [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Create a conda environment. Use the following command to create a new environment named "sfl" (replace "sfl" with your desired environment name):

        conda env create -f environment.yml --name sfl 

3. Clone the [FMMECG3D repository](https://github.com/FMMGroupVa/FMMECG3D/tree/main) [1]  by running:

        git clone git@github.com:FMMGroupVa/FMMECG3D.git <your_custom_fmm_r_repository_path>

### Setting up the configuration
We use [hydra](https://hydra.cc/docs/intro/) to handle configuration file. Go to [the configuration file](/conf/ecg_anomaly_detection.yaml) and modify the ??? fields with your own paths. 

### Running your script
1. Either download the [preprocessed dataset](TODO: add fmm shared dataset file shere) (**strongly suggested**) or run the generate_ptb_xl_fmm_ds file (could take days depending on your computational power capabilities)
2. Unzip the file in your data folder (that you specified in the configuration file)
3. Run the [example](example.ipynb) notebook. You can speficy additional arguments in the overrides arguments of the hydra compose function in the first cell -->

## Citation

If you use this code in your work, please cite our paper [2]:

        @misc{verardo2023fmmhead,
        title         = {FMM-Head: Enhancing Autoencoder-based ECG anomaly detection with prior knowledge}, 
        author        = {Giacomo Verardo and Magnus Boman and Samuel Bruchfeld 
                        and Marco Chiesa and Sabine Koch and Gerald Q. Maguire Jr. 
                        and Dejan Kostic},
        year          = {2023},
        eprint        = {2310.05848},
        archivePrefix = {arXiv},
        primaryClass  = {cs.LG}
        }



### References
[1]: Rueda, C., Rodríguez-Collado, A., Fernández, I., Canedo, C., Ugarte, M. D., & Larriba, Y. (2022). A unique cardiac electrocardiographic 3D model. Toward interpretable AI diagnosis. *iScience*, 25(12), 105617. [DOI: 10.1016/j.isci.2022.105617](https://doi.org/10.1016/j.isci.2022.105617)

[2]: Giacomo Verardo, Magnus Boman, Samuel Bruchfeld, Marco Chiesa, Sabine Koch, Gerald Q. Maguire Jr., Dejan Kostic. (2023). FMM-Head: Enhancing Autoencoder-based ECG anomaly detection with prior knowledge - arXiv:2310.05848 [cs.LG]

