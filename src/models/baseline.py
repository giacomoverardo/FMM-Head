import numpy as np
import sklearn
from typing import Dict, List, Tuple
from src.plot.general import plot_roc

class BaselineModel:
    def __init__(self, sequence_length:int, num_features:int, name:str, num_warmup_epochs:int,**kwargs):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.name = name
        self.num_warmup_epochs = num_warmup_epochs
        self.need_warmup = True if num_warmup_epochs>0 else False
    
    def fit(self, dataset, epochs, validation_data=None, callbacks=None):
        # Train the model
        print(f"Warning: {self.name} model does not have a fit method.")  
        print("Instead, the original training code will be launched.")  
        print("Callbacks and validation will not be considered") 


    def compile(self, *args, **kwargs):
        # Compile the model (for instance, with an optimizer)
        print(f"Warning: {self.name} model does not have a compile method.")
        
    def load_weights(self, path):
        # Load model weights from path
        raise NotImplementedError
    
    def expect_partial(self):
        pass
    
    def compute_roc(self, data_dict:Dict, normal_class:int, filename:str)->Dict:
        """Compute roc curve and area under the roc curve

        Args:
            data_dict (Dict): input data dictionary
            normal_class (int): index of the normal class
            filename (str): file path where to save the output dictionary

        Returns:
            Dict: dictionary containing roc curve and area under the roc
        """
        y_scores = self.predict_loss(data_dict)
        labels = data_dict["labels"]
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels!=normal_class, y_score=y_scores)
        roc_auc = sklearn.metrics.auc(fpr,tpr)
        plot_roc(fpr, tpr, roc_auc, filename)
        roc_dict = { "fpr":fpr.tolist(), "tpr":tpr.tolist(), "thresholds":thresholds.tolist(),"roc_auc":roc_auc}
        return roc_dict
    
    def compute_class_loss(self, data_dict:Dict, classes:List)->Dict:
        """Compute average and std value for each dataset class 

        Args:
            data_dict (Dict): input data dictionary
            classes (List): classes names

        Returns:
            Dict: Mean and standard deviation for each class
        """
        labels = data_dict["labels"]
        loss = self.predict_loss(data_dict)
        class_loss_dict = {}
        class_mean_loss = np.zeros_like(classes, dtype=np.float32)
        class_std_loss = np.zeros_like(classes, dtype=np.float32)
        for i,class_name in enumerate(classes):
            loss_class = loss[labels==i]
            class_mean_loss[i] = np.mean(loss_class,axis=0)
            class_std_loss[i] = np.std(loss_class,axis=0)
        class_loss_dict = {"mean":class_mean_loss.tolist(), "std":class_std_loss.tolist()}
        return class_loss_dict

    def get_confusion_matrix(self, data_dict:Dict, threshold:float, normal_class:int)->np.ndarray:
        """Generate confusion matrix from model given the threshold

        Args:
            data_dict (Dict): input data dictionary
            threshold (float): threshold value
            normal_class (int): index of the normal class

        Returns:
            np.ndarray: the confusion matrix
        """
        labels = data_dict["labels"]
        loss = self.predict_loss(data_dict)
        real_abnormal_indexes = labels!=normal_class
        detected_abnormal_indexes = loss>threshold
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true = real_abnormal_indexes,
                                                    y_pred=detected_abnormal_indexes)
        return confusion_matrix
    
    def predict_loss(self, data_dict: Dict)->np.ndarray:
        """Predict the loss for each sample in data dict

        Args:
            data_dict (Dict): input data dictionary

        Raises:
            NotImplementedError: implement the function in children classes
        
        Returns:
            np.ndarray: array with one loss per inputs sample
        """
        raise NotImplementedError
    
    def predict(self, data_dict:Dict, *args, **kwargs)->np.ndarray:
        """Predict function

        Args:
            data_dict (Dict): input dictionary

        Returns:
            np.ndarray: prediction output
        """
        raise NotImplementedError

    def test_step(self, *args, **kwargs):
        # Test the model (for instance, with an optimizer)
        print(f"Warning: {self.name} model does not have a test method.")

    @property
    def trainable_weights(self):
        # Collect trainable weights
        raise NotImplementedError

    @property
    def non_trainable_weights(self):
        # Collect non trainable weights
        raise NotImplementedError
    
    def get_model_size(self, path:str)->int:
        """Get model size in bytes

        Args:
            path (str): path containing the files with the model weigths

        Raises:
            NotImplementedError: implement the method in children classes

        Returns:
            int: size of the model in number of bytes
        """
        raise NotImplementedError

if __name__=='__main__':
    pass


if __name__ == '__main__':
    pass