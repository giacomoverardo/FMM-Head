import os
from typing import Dict, List
import torch
from tqdm import tqdm
# from src.baselines.diffusionae.src.eval import evaluate
# from src.baselines.diffusionae.src.parser import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
# from src.baselines.diffusionae.src.my_plotting import plotter
import numpy as np
from src.models.baseline import BaselineModel
from src.baselines.diffusionae.models2 import Autoencoder_Diffusion, TransformerBasicBottleneckScaling, \
                                            TransformerBasicv2Scaling, ConditionalDiffusionTrainingNetwork
import tensorflow as tf
from src.utils.nn import get_torch_nn_parameters

device = 'cuda'

def convert_to_windows(data, n_window):
    windows = list(torch.split(data, n_window))
    for i in range (n_window-windows[-1].shape[0]):
        windows[-1] = torch.cat((windows[-1], windows[-1][-1].unsqueeze(0)))
    return torch.stack(windows)

# parametri: point_global, point_contextual etc
def load_dataset(dataset, part=None):
    loader = [] 
    folder = 'DiffusionAE/processed/' + dataset

    for file in ['train', 'test', 'validation', 'labels', 'labels_validation']:
        if part is None:
            loader.append(np.load(os.path.join(folder, f'{file}.npy')))
        else:
            loader.append(np.load(os.path.join(folder, f'{part}_{file}.npy')))
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    validation_loader = DataLoader(loader[2], batch_size=loader[2].shape[0])
    return train_loader, test_loader, validation_loader, loader[3], loader[4]

def load_model(training_mode, lr, window_size, p1, p2, dims, batch_size, noise_steps, denoise_steps, model_name, optimizer):
    scheduler=None	
    model = None
    diffusion_training_net = ConditionalDiffusionTrainingNetwork(dims, int(window_size), batch_size, noise_steps, denoise_steps).float()
    diffusion_prediction_net = ConditionalDiffusionTrainingNetwork(dims, int(window_size), batch_size, noise_steps, denoise_steps, train=False).float()
    if training_mode == 'both':
        if model_name == 'Autoencoder_Diffusion':
            model = Autoencoder_Diffusion(dims, float(lr), int(window_size), p1, p2).float()
        elif model_name == 'TransformerBasicBottleneckScaling': 
            model = TransformerBasicBottleneckScaling(dims, float(lr), int(window_size), batch_size).float()
        else:
            model = TransformerBasicv2Scaling(dims, float(lr), int(window_size)).float()
        # optimizer = torch.optim.Adam(list(model.parameters()) + list(diffusion_training_net.parameters()), lr=model.lr)
        optimizer.add_param_group({'params': list(model.parameters()) + list(diffusion_training_net.parameters())})
    else:
        # optimizer = torch.optim.Adam(diffusion_training_net.parameters(), lr=float(lr))
        optimizer.add_param_group({'params': diffusion_training_net.parameters()})
    return model, diffusion_training_net, diffusion_prediction_net, optimizer, scheduler

# CHECKPOINT_FOLDER = '/home/giacomo/otherRepos/DiffusionAE/wandb/model'
def save_model(model, experiment, diffusion_training_net, optimizer, scheduler, anomaly_score, epoch, diff_loss, ae_loss, folder):
    # folder = f'{checkpoint_folder}/{experiment}/'
    os.makedirs(folder, exist_ok=True)
    if model:
        file_path_model = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'ae_loss': ae_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),}, file_path_model)
    file_path_diffusion = f'{folder}/diffusion.ckpt'
    torch.save({
        'epoch': epoch,
        'diffusion_loss': diff_loss,
        'model_state_dict': diffusion_training_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),}, file_path_diffusion)
    print('saved model at ' + folder)

def load_from_checkpoint(training_mode, experiment, model, diffusion_training_net, folder):
    # folder = f'{checkpoint_folder}/{experiment}'
    file_path_model = f'{folder}/model.ckpt'
    file_path_diffusion = f'{folder}/diffusion.ckpt'
    # load model
    if training_mode == 'both':
        checkpoint_model = torch.load(file_path_model)
        model.load_state_dict(checkpoint_model['model_state_dict'])
    else: 
        model = None
    # load diffusion
    checkpoint_diffusion = torch.load(file_path_diffusion)
    diffusion_training_net.load_state_dict(checkpoint_diffusion['model_state_dict'])
    return model, diffusion_training_net

def get_diffusion_sample(diffusion_prediction_net, conditioner, k):
    if k <= 1:
        return diffusion_prediction_net(conditioner)
    else:  
        diff_samples = []
        for _ in range(k):
            diff_samples.append(diffusion_prediction_net(conditioner))
        return torch.mean(torch.stack(diff_samples), axis = 0)

def backprop(epoch, model, diffusion_training_net, diffusion_prediction_net, data, diff_lambda, optimizer, 
             scheduler, training_mode, anomaly_score, k, training = True, folder = None, model_name=None):
    e = epoch
    feats = 1
    l = nn.MSELoss(reduction = 'none')
    data_x = torch.tensor(data, dtype=torch.float32); dataset = TensorDataset(data_x, data_x)
    bs = diffusion_training_net.batch_size if not model else model.batch
    dataloader = DataLoader(dataset, batch_size = bs)
    w_size = diffusion_training_net.window_size
    l1s, diff_losses, ae_losses = [], [], []
    samples = []
    if training:
        if training_mode == 'both':
            model.train()
        diffusion_training_net.train()
        for d, _ in dataloader:
            if model_name == 'Autoencoder_Diffusion':
                local_bs = d.shape[0]
                window = d.view(local_bs, -1)
            else:
                window = d
            window = window.to(device)
            if training_mode == 'both':
                if model_name == 'Autoencoder_Diffusion':
                    ae_reconstruction = model(window)
                else:
                    ae_reconstruction = model(window, window)
                # B x (feats * win)
                ae_loss = l(ae_reconstruction, window)
                ae_reconstruction = ae_reconstruction.reshape(-1, w_size, feats)
                # un tensor cu un element
                diffusion_loss, _ = diffusion_training_net(ae_reconstruction)
                ae_losses.append(torch.mean(ae_loss).item())
                diff_losses.append(torch.mean(diffusion_loss).item())
                if e < 5:
                    loss = torch.mean(ae_loss)
                else:
                    loss = diff_lambda * diffusion_loss + torch.mean(ae_loss)
            else:
                # diff only
                window = window.reshape(-1, w_size, feats)
                loss, _ = diffusion_training_net(window)
            l1s.append(loss.item())
            optimizer.zero_grad()
            loss.backward()                                                                                                     
            optimizer.step()
        tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
        tqdm.write(f'Epoch {epoch},\tAE = {np.mean(ae_losses)}')
        tqdm.write(f'Epoch {epoch},\tDiff = {np.mean(diff_losses)}')
        return np.mean(l1s), np.mean(ae_losses), np.mean(diff_losses)
    else:
        with torch.no_grad():
            if training_mode == 'both':
                model.eval()
            diffusion_prediction_net.load_state_dict(diffusion_training_net.state_dict())
            diffusion_prediction_net.eval()
            diffusion_training_net.eval()
            l1s = [] # scores
            sum_losses = []
            ae_losses = []
            diff_losses = []
            recons = []
            for d, _ in dataloader:
                if model_name == 'Autoencoder_Diffusion':
                    local_bs = d.shape[0]
                    window = d.view(local_bs, -1)
                else:
                    window = d
                window = window.to(device)
                window_reshaped = window.reshape(-1, w_size, feats)
                if training_mode == 'both':
                    if model_name == 'Autoencoder_Diffusion':
                        ae_reconstruction = model(window)
                    else:
                        ae_reconstruction = model(window, window)
                    ae_reconstruction_reshaped = ae_reconstruction.reshape(-1, w_size, feats)
                    recons.append(ae_reconstruction_reshaped)
                    ae_loss = l(ae_reconstruction, window)
                    ae_losses.append(torch.mean(ae_loss).item())
                    _, diff_sample = diffusion_prediction_net(ae_reconstruction_reshaped)
                    diff_sample = torch.squeeze(diff_sample, 1)
                    diffusion_loss = l(diff_sample, window_reshaped)
                    diffusion_loss = torch.mean(diffusion_loss).item()
                    sum_losses.append(torch.mean(ae_loss).item() + diffusion_loss)
                    diff_losses.append(diffusion_loss)
                    samples.append(diff_sample)
                    if anomaly_score == 'both': # 1
                        loss = l(diff_sample, ae_reconstruction_reshaped)
                    elif anomaly_score == 'diffusion': # 3
                        loss = l(diff_sample, window_reshaped)
                    elif anomaly_score == 'autoencoder': # 2
                        loss = l(ae_reconstruction, window)
                    elif anomaly_score == 'sum': # 4 = 2 + 3
                        loss = l(ae_reconstruction, window) + l(window, diff_sample)
                    elif anomaly_score == 'sum2': # 5 = 1 + 2
                        loss = l(diff_sample, ae_reconstruction) + l(ae_reconstruction, window)
                    elif anomaly_score == 'diffusion2': # 6 - 3 conditionat de gt
                        diff_sample = get_diffusion_sample(diffusion_prediction_net, window_reshaped, k)
                        loss = l(diff_sample, window_reshaped)
                else:
                    _, x_recon = diffusion_prediction_net(window_reshaped)
                    x_recon = torch.squeeze(x_recon, 1)
                    samples.append(x_recon)
                    loss = l(x_recon, window_reshaped)
                l1s.append(loss)
        if training_mode == 'both':
            return torch.cat(l1s).detach().cpu().numpy(), np.mean(sum_losses), np.mean(ae_losses), np.mean(diff_losses), torch.cat(samples).detach().cpu().numpy(), torch.cat(recons).detach().cpu().numpy()
        return torch.cat(l1s).detach().cpu().numpy(), np.mean(sum_losses), np.mean(ae_losses), np.mean(diff_losses), torch.cat(samples).detach().cpu().numpy()

# if __name__ == '__main__':
#     pass
#     # TEST ON TEST SET
#     #load model from checkpoint
#     model, diffusion_training_net, diffusion_prediction_net, optimizer, scheduler = \
#                         load_model(training_mode ,args.lr, args.window_size, args.p1, args.p2, labels.shape[1], args.batch_size, args.noise_steps, args.denoise_steps)
#     model, diffusion_training_net = load_from_checkpoint(training_mode, experiment, model, diffusion_training_net, self.save_folder)
#     if model:
#         model = model.to(device)
            
#     diffusion_training_net = diffusion_training_net.to(device)
#     diffusion_prediction_net = diffusion_prediction_net.to(device)
#     # pass test set through the model
#     if model:
#         if args.test_only:
#         #test again on val for double check + get best thresh on validation set to use for test
#             loss0, val_loss, ae_loss_val, diff_loss_val, samples, recons = backprop(e, model, diffusion_training_net, diffusion_prediction_net, validationD, args.diff_lambda, optimizer, scheduler, training_mode, args.anomaly_score, args.k, training=False)
#             loss0 = loss0.reshape(-1,feats)

#             lossFinal = np.mean(np.array(loss0), axis=1)
#             # lossFinal = np.max(np.array(loss0), axis=1)
#             labelsFinal = (np.sum(validation_labels, axis=1) >= 1) + 0

#             result, fprs, tprs = evaluate(lossFinal, labelsFinal)
#             validation_thresh = result['threshold']
#             result_roc = result["ROC/AUC"]
#             result_f1 = result["f1"]
#             wandb.run.summary["f1_val"] = result_f1
#             wandb.run.summary["roc_val"] = result_roc
#             wandb.run.summary["f1_pa_val"] = result['f1_max'] 
#             wandb.run.summary["roc_pa_val"] = result['roc_max']
#             wandb.run.summary["val_loss"] = val_loss
#             wandb.run.summary["ae_loss_val"] = ae_loss_val
#             wandb.run.summary["diff_loss_val"] = diff_loss_val

#             # for dim in range(0, feats):
#             #     fig = plotter(f'{experiment}_VAL', args.anomaly_score, validationD.reshape(-1, feats), lossFinal, labelsFinal, result, recons.reshape(-1, feats), samples.reshape(-1, feats), None, dim=dim, plot_test=True, epoch=e)

#         loss0, test_loss, ae_loss_test, diff_loss_test, samples, recons = backprop(e, model, diffusion_training_net, diffusion_prediction_net, testD, args.diff_lambda, optimizer, scheduler, training_mode, args.anomaly_score, args.k, training=False)
#         loss0 = loss0.reshape(-1,feats)

#         lossFinal = np.mean(np.array(loss0), axis=1)
#         np.save(os.path.join(savepath,f'{args.dataset}_{args.anomaly_score}_score_scores.npy'), lossFinal)
#         np.save(os.path.join(savepath,f'{args.dataset}_{args.anomaly_score}_score_recons.npy'), samples)
#         # np.save('/root/Diff-Anomaly/TranAD/plots_for_paper/shapelet_scores_for_example.npy', lossFinal)
#         # lossFinal = np.max(np.array(loss0), axis=1)
#         labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
#         #validation_thresh = 0.0019
#         result, fprs, tprs = evaluate(lossFinal, labelsFinal, validation_thresh=validation_thresh)
#         result_roc = result["ROC/AUC"]
#         result_f1 = result["f1"]
#         wandb.run.summary["f1_test"] = result_f1
#         wandb.run.summary["roc_test"] = result_roc
#         wandb.run.summary["f1_pa_test"] = result['f1_max'] 
#         #wandb.run.summary["roc_pa_test"] = result['roc_max']
#         wandb.run.summary["test_loss"] = test_loss
#         wandb.run.summary["ae_loss_test"] = ae_loss_test
#         wandb.run.summary["diff_loss_test"] = diff_loss_test
#         wandb.run.summary["validation_thresh"] = validation_thresh

#         #for dim in range(0, feats):
#         #    fig = plotter(f'{experiment}_TEST', args.anomaly_score, testD.reshape(-1, feats), lossFinal, labelsFinal, result, recons.reshape(-1, feats), samples.reshape(-1, feats), None, dim=dim, plot_test=True, epoch=e)
        
#     else: 
#         if args.test_only:
#             loss0, _, _, val_loss, samples = backprop(e, model, diffusion_training_net, diffusion_prediction_net, validationD, args.diff_lambda, optimizer, scheduler, training_mode, args.anomaly_score, args.k, training=False)
#             loss0 = loss0.reshape(-1,feats)

#             lossFinal = np.mean(np.array(loss0), axis=1)
#             labelsFinal = (np.sum(validation_labels, axis=1) >= 1) + 0

#             result, fprs, tprs = evaluate(lossFinal, labelsFinal)
#             result_roc = result["ROC/AUC"]
#             result_f1 = result["f1"]
#             validation_thresh = result['threshold']
#             wandb.run.summary["f1_val"] = result_f1
#             wandb.run.summary["roc_val"] = result_roc
#             wandb.run.summary["f1_pa_val"] = result['f1_max'] 
#             wandb.run.summary["val_loss"] = val_loss
#             wandb.run.summary["validation_thresh"] = validation_thresh
#             #for dim in range(0, feats):
#             #    plotter(f'{experiment}_VAL', args.dataset, validationD.reshape(-1, feats), lossFinal, labelsFinal, result, None, samples.reshape(-1, feats), None, dim=dim, plot_test=True, epoch=e)
#         loss0, _, _, test_loss, samples = backprop(e, model, diffusion_training_net, diffusion_prediction_net, testD, args.diff_lambda, optimizer, scheduler, training_mode, args.anomaly_score, args.k, training=False)
#         loss0 = loss0.reshape(-1,feats)

#         lossFinal = np.mean(np.array(loss0), axis=1)
#         np.save(os.path.join(savepath,f'{args.dataset}_diff_only_scores.npy'), lossFinal)
#         np.save(os.path.join(savepath,f'{args.dataset}_diff_only_recons.npy'), samples)
        
#         labelsFinal = (np.sum(labels, axis=1) >= 1) + 0

#         result = evaluate(lossFinal, labelsFinal, validation_thresh=validation_thresh)
#         result_roc = result["ROC/AUC"]
#         result_f1 = result["f1"]
#         #for dim in range(0, feats):
#         #    plotter(f'{experiment}_TEST', args.dataset, testD.reshape(-1, feats), lossFinal, labelsFinal, result, None, samples.reshape(-1, feats), None, dim=dim, plot_test=True, epoch=e)
#         wandb.run.summary["f1_test"] = result_f1
#         wandb.run.summary["roc_test" ] = result_roc
#         wandb.run.summary["f1_pa_test"] = result['f1_max'] 
#         wandb.run.summary["roc_pa_test"] = result['roc_max']
#         wandb.run.summary["test_loss"] = test_loss
    
    # wandb.finish()  

class Diffusion_AE(BaselineModel):
    def __init__(self, sequence_length: int, num_features: int, name: str, num_warmup_epochs: int, 
                diff_lambda: float, denoise_steps: int, anomaly_score: str, training: str, model,
                window_size: int, noise_steps: int, num_epochs: int, optimizer,
                learning_rate: float, batch_size: int, p1:int, p2:int, k:int, v:bool, test_only:bool, **kwargs):
        super().__init__(sequence_length, num_features, name, num_warmup_epochs, **kwargs)
        self.diff_lambda = diff_lambda
        self.denoise_steps = denoise_steps
        self.anomaly_score = anomaly_score
        self.training = training
        self.model_name = model
        self.window_size = window_size
        self.noise_steps = noise_steps
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.p1 = p1
        self.p2 = p2
        self.k = k
        self.v = v
        self.test_only = test_only
        self.create_model()

    def fit(self, dataset, epochs, validation_data=None, callbacks=None):
        super().fit(dataset, epochs, validation_data, callbacks)
        checkpoint_callback: tf.keras.callbacks.ModelCheckpoint = callbacks[2]
        self.save_folder = os.path.dirname(checkpoint_callback.filepath) #os.path.join(os.path.dirname(checkpoint_callback.filepath),"ecg-adgan")
        input_data = dataset["inputs"]
        self.train(input_data)
        callbacks[4].epoch_times = self.epoch_times
        # Custom object to provide a history-like output
        class Myobject:
            pass
        history = Myobject()
        history.history = self.progress
        return history

    def save_models(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_folder,"model"))
        torch.save(self.diffusion_training_net.state_dict(), os.path.join(self.save_folder,"diffusion_training_net"))
        torch.save(self.diffusion_prediction_net.state_dict(), os.path.join(self.save_folder,"diffusion_prediction_net"))

    def load_weights(self, path):
        self.save_folder = os.path.dirname(path)
        self.model.load_state_dict(torch.load(os.path.join(self.save_folder,"model")))
        self.diffusion_training_net.load_state_dict(torch.load(os.path.join(self.save_folder,"diffusion_training_net")))
        self.diffusion_prediction_net.load_state_dict(torch.load(os.path.join(self.save_folder,"diffusion_prediction_net")))
        return self

    def create_model(self):
        self.training_mode = 'both' if not self.training else self.training
        self.anomaly_score = None if not self.anomaly_score else self.anomaly_score
        self.model, self.diffusion_training_net, \
        self.diffusion_prediction_net, self.optimizer, self.scheduler = \
                            load_model(training_mode=self.training_mode , lr=self.learning_rate, window_size=self.window_size, p1=self.p1, p2=self.p2, dims=self.num_features, 
                                       batch_size=self.batch_size, noise_steps=self.noise_steps, denoise_steps=self.denoise_steps, optimizer=self.optimizer, model_name=self.model_name)
        if self.model:
            self.model = self.model.to(device)
        self.diffusion_training_net = self.diffusion_training_net.to(device)
        self.diffusion_prediction_net = self.diffusion_prediction_net.to(device)

    # def compute_class_loss(self, data_dict: Dict, classes: List) -> Dict:
    #     labels = data_dict["labels"]
    #     loss_samples = backprop(0, self.model, self.diffusion_training_net, self.diffusion_prediction_net, data_dict["inputs"], 
    #                                             self.diff_lambda, self.optimizer, self.scheduler, self.training_mode, self.anomaly_score, 
    #                                             self.k, model_name=self.model_name, training=False)[0]
    #     loss = np.average(np.average(loss_samples,axis=-1),axis=-1)
    #     class_loss_dict = {}
    #     class_mean_loss = np.zeros_like(classes, dtype=np.float32)
    #     class_std_loss = np.zeros_like(classes, dtype=np.float32)
    #     for i,class_name in enumerate(classes):
    #         loss_class = loss[labels==i]
    #         class_mean_loss[i] = np.mean(loss_class,axis=0)
    #         class_std_loss[i] = np.std(loss_class,axis=0)
    #     class_loss_dict = {"mean":class_mean_loss.tolist(), "std":class_std_loss.tolist()}
    #     return class_loss_dict
    
    def predict_loss(self, data_dict: Dict, load_from_saved:bool=True) -> np.ndarray:
        # if(hasattr(self, 'saved_loss') and load_from_saved):
        #     print("Loading loss values from previous computations to save time.\n\
        #         To compute the loss, run method with argument load_from_saved=False")
        #     loss_samples = self.saved_loss
        # else:
        loss_samples = backprop(0, self.model, self.diffusion_training_net, self.diffusion_prediction_net, data_dict["inputs"], 
                                    self.diff_lambda, self.optimizer, self.scheduler, self.training_mode, self.anomaly_score, 
                                    self.k, model_name=self.model_name, training=False)[0]
            # self.saved_loss = loss_samples
        loss = np.average(np.average(loss_samples,axis=-1),axis=-1)
        return loss
    
    def predict(self, data_dict: Dict, *args, **kwargs) -> np.ndarray:
        return self.predict_loss(data_dict, load_from_saved=False)

    def get_model_size(self, path: str) -> int:
        model_size = os.path.getsize(os.path.join(self.save_folder,"model"))
        model_size += os.path.getsize(os.path.join(self.save_folder,"diffusion_training_net"))
        return model_size
    
    def train(self, input_data):
        savepath = self.save_folder
        anomaly_scores = [self.anomaly_score]

        if self.training == 'diffusion':
            experiment = 'diffv4'
        elif self.model_name == 'Autoencoder_Diffusion':
            experiment = 'autoencoder_both'
        elif self.model_name == 'TransformerBasicBottleneckScaling':
            experiment = 'tr_bn_diffv4'
        else:
            experiment = 'tr_basic_diffv4'
        
        # experiment += f'_{args.dataset}_{self.noise_steps}-{self.denoise_steps}_{self.diff_lambda}_{self.learning_rate}_{self.batch_size}_{self.window_size}'

        # if self.training == 'both':
        #     experiment += f'_{anomaly_scores[0]}_score' 
        #experiment += f'_{args.id}'   

        # wandb.init(project="anomaly-mts", entity="giacomoverardo", config=config, group=args.group)
        # wandb.run.name = experiment
        
        # dataset_name = args.dataset
        # part = None if not args.file else args.
        part = None
        window_size = int(self.window_size)
        synthetic_datasets = ['point_global', 'point_contextual', 'pattern_shapelet', 'pattern_seasonal', 'pattern_trend', 'all_types', 'pattern_trendv2']
        
        # train_loader, test_loader, validation_loader, labels, validation_labels = load_dataset(dataset_name, part)
        train_loader = DataLoader(input_data, batch_size=self.batch_size)

        # trainD, testD, validationD = next(iter(train_loader)), next(iter(test_loader)), next(iter(validation_loader))
        trainD = next(iter(train_loader))
        # trainO, testO, validationO = trainD, testD, validationD
        trainO = trainD
        if self.v:
            print(f'\ntrainD.shape: {trainD.shape}')
            # print(f'testD.shape: {testD.shape}')
            # print(f'validationD.shape: {validationD.shape}')
            # print(f'labels.shape: {labels.shape}')
        
        # feats=labels.shape[1]    
        feats = input_data.shape[-1]  
        # trainD, testD, validationD = convert_to_windows(trainD, window_size), convert_to_windows(testD, window_size), convert_to_windows(validationD, window_size)
        epoch = -1

        e = epoch + 1; start = time()
        max_roc_scores = [[0, 0, 0]] * 6
        max_f1_scores = [[0, 0, 0]] * 6
        roc_scores = []
        f1_scores = []
        f1_max = 0
        roc_max = 0
        validation_thresh = 0

        self.progress = {'sum_loss_train': [],
                    'ae_loss_train': [],
                    'diff_loss_train': []}
        # anomaly_scores = ['diffusion']
        #alpha = 0
        self.epoch_times = []
        if not self.test_only:
            for e in tqdm(list(range(epoch+1, epoch+self.num_epochs+1))):
                start_epoch_time = time()
                train_loss, ae_loss, diff_loss = backprop(e, self.model, self.diffusion_training_net, self.diffusion_prediction_net, trainD, 
                                                          self.diff_lambda, self.optimizer, self.scheduler, self.training_mode, self.anomaly_score, 
                                                          self.k, model_name=self.model_name)
                end_epoch_time = time()
                self.progress['sum_loss_train'].append((train_loss))
                self.progress['ae_loss_train'].append(ae_loss)
                self.progress['diff_loss_train'].append((diff_loss))
                self.save_models()
                self.epoch_times.append(end_epoch_time-start_epoch_time)
                # wandb.log({
                #     'sum_loss_train': train_loss,
                #     'ae_loss_train': ae_loss,
                #     'diff_loss_train': diff_loss,
                #     'epoch': e
                # }, step=e)
                # if training_mode == 'both':
                #     for idx, a_score in enumerate(anomaly_scores):
                #         if ae_loss + diff_loss < 0.15:
                #             loss0, val_loss, ae_loss_val, diff_loss_val, samples, recons = backprop(e, model, diffusion_training_net, diffusion_prediction_net, validationD, args.diff_lambda, optimizer, scheduler, training_mode, a_score, args.k, training=False)
                #             if idx == 0:    
                #                 wandb.log({
                #                     'sum_loss_val': val_loss,
                #                     'ae_loss_val': ae_loss_val,
                #                     'diff_loss_val': diff_loss_val,
                #                     'epoch': e
                #                 }, step=e)
                #             loss0 = loss0.reshape(-1,feats)

                #             lossFinal = np.mean(np.array(loss0), axis=1)
                #             labelsFinal = (np.sum(validation_labels, axis=1) >= 1) + 0

                #             result, fprs, tprs = evaluate(lossFinal, labelsFinal)
                #             result_roc = result["ROC/AUC"]
                #             result_f1 = result["f1"]
                #             wandb.log({'roc': result_roc, 'f1': result_f1}, step=e)
                #             if result_roc > max_roc_scores[idx][0]:
                #                 max_roc_scores[idx] = [result_roc, result_f1, e]
                #                 wandb.run.summary["f1_for_best_roc"] = result_f1
                #                 wandb.run.summary["best_roc"] = result_roc
                #                 wandb.run.summary["best_roc_epoch"] = e
                #             if result_f1 > max_f1_scores[idx][1]:
                #                 max_f1_scores[idx] = [result_roc, result_f1, e]
                #                 save_model(model, experiment, diffusion_training_net, optimizer, None, a_score, e, diff_loss, ae_loss, self.save_folder)
                #                 validation_thresh = result['threshold']
                #                 wandb.run.summary["best_f1"] = result_f1
                #                 wandb.run.summary["roc_for_best_f1"] = result_roc
                #                 wandb.run.summary["best_roc_epoch"] = e
                #                 wandb.run.summary["best_f1_epoch"] = e
                #                 wandb.run.summary["f1_pa"] = result['f1_max'] 
                #                 wandb.run.summary["roc_pa"] = result['roc_max']
                #             if e % 5 == 0: 
                #                 for dim in range(0, feats):
                #                     fig = plotter(experiment, a_score, validationD.reshape(-1, feats), lossFinal, labelsFinal, os.path.join(savepath, "plot"),
                #                                 result, recons.reshape(-1, feats), samples.reshape(-1, feats), None, dim=dim, plot_test=True, epoch=e)
                #             if args.v:
                #                 print(str(e) + ' ROC: ' + str(result_roc) + ' F1: ' + str(result_f1) + '\n')
                        
                # else:
                #     if train_loss < 0.15:
                #         loss0, _, _, val_loss, samples = backprop(e, model, diffusion_training_net, diffusion_prediction_net, validationD, args.diff_lambda, optimizer, scheduler, training_mode, args.anomaly_score, args.k, training=False)
                #         wandb.log({'val_loss': loss0.mean(), 'epoch': e}, step=e)
                #         loss0 = loss0.reshape(-1,feats)
                #         lossFinal = np.mean(np.array(loss0), axis=1)
                #         labelsFinal = (np.sum(validation_labels, axis=1) >= 1) + 0
                #         result, fprs, tprs = evaluate(lossFinal, labelsFinal)
                #         result_roc = result["ROC/AUC"]
                #         result_f1 = result["f1"]
                #         wandb.log({'roc': result_roc, 'f1': result_f1}, step=e)
                #         if result_f1 > f1_max:
                #             save_model(None, experiment, diffusion_prediction_net, optimizer, None, -1, e, train_loss, None, self.save_folder)
                #             f1_max = result_f1
                #             validation_thresh = result['threshold']
                #             wandb.run.summary["best_f1"] = f1_max
                #             wandb.run.summary["roc_for_best_f1"] = result_roc
                #             wandb.run.summary["best_f1_epoch"] = e
                #             wandb.run.summary["validation_thresh"] = validation_thresh
                #         if result_roc > roc_max:
                #             roc_max = result_roc 
                #             wandb.run.summary["f1_for_best_roc"] = result_f1
                #             wandb.run.summary["best_roc"] = roc_max
                #             wandb.run.summary["best_roc_epoch"] = e
                #         wandb.log({'roc': result_roc, 'f1': result_f1}, step=e)
                #         if e % 5 == 0:
                #             for dim in range(0, feats):
                #                 plotter(f'{experiment}_VAL', args.dataset, validationD.reshape(-1, feats), lossFinal, labelsFinal, os.path.join(savepath, "plot"),
                #                         result, None, samples.reshape(-1, feats), None, dim=dim, plot_test=True, epoch=e)
                #         if args.v:
                #             print(f"testing loss #{e}: {loss0.mean()}")
                #             # print(f"training loss #{e}: {loss1.mean()}")
                #             print(f"final ROC #{e}: {result_roc}")
                #             print(f"F1 #{e}: {result_f1}")
    
    def get_number_parameters(self, p_type):
        model_pars = get_torch_nn_parameters(self.model, p_type)
        diff_pars = get_torch_nn_parameters(self.diffusion_training_net, p_type)
        return model_pars+diff_pars
    
    def get_number_trainable_parameters(self): 
        return self.get_number_parameters("trainable")
    
    def get_number_non_trainable_parameters(self): 
        return self.get_number_parameters("non-trainable")

if __name__=='__main__':
    pass