from src.models.baseAE import BaseAE
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Masking, RepeatVector, \
                                    Flatten, TimeDistributed, LSTMCell, Identity, \
                                    Dropout, GlobalAveragePooling1D, ReLU
from tensorflow.keras.models import Sequential
# from src.models.fmm_ae import FMM_AE_regression_circular_ang, FMM_head_ang
import tensorflow as tf
from src.models.baseFMMAE import Base_FMM_AE
from src.models.nnmodels import get_dense_network
class Dense_ae(BaseAE):
    def __init__(self, units, sequence_length, num_features, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = len(units)
        self.dropout_rate = dropout_rate
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.sequential = get_dense_network(numNodesPerLayer=units,dropoutRate=dropout_rate,maskValue=0.0,addInFlatten=True)
        self.sequential.add(Dense(sequence_length))

    def call(self, inputs, training):
        x = inputs["inputs"]
        x = self.sequential(x)
        x = x[..., tf.newaxis]
        return {"data":inputs["inputs"], "predicted_data":x}

class FMM_dense_ae(Base_FMM_AE):
    def __init__(self, units, sequence_length, num_features, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = len(units)
        self.encoder = get_dense_network(numNodesPerLayer=units,dropoutRate=dropout_rate,maskValue=0.0,addInFlatten=True)
        self.global_avg_pooling = Identity()
        self.dropout_rate = dropout_rate
        self.sequence_length = sequence_length
        self.num_features = num_features

if __name__ == '__main__':
    pass