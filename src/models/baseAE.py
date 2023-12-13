import tensorflow as tf
from tensorflow import keras
from src.utils.metrics import masked_mse

class BaseAE(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.need_warmup = False
        # Metrics:
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        # Test metrics
        self.test_total_loss_tracker = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.test_total_loss_tracker,
        ]
        
    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss_dict = self.compute_loss(self(inputs=data,training=True))
            total_loss_mean = tf.reduce_mean(loss_dict["total_loss"])
        grads = tape.gradient(total_loss_mean, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss_dict["total_loss"])
        return {
            "loss": self.total_loss_tracker.result()}
        
    def test_step(self, data):
        loss_dict = self.compute_loss(self(inputs=data,training=False))
        self.test_total_loss_tracker.update_state(loss_dict["total_loss"])
        return {
            "loss": self.test_total_loss_tracker.result(),
        }
        
    def compute_loss(self, inputs):
        data = inputs["data"]
        predicted_data = inputs["predicted_data"]
        loss = masked_mse(data,predicted_data)
        return {"total_loss":loss}
    
    def call(self, inputs, training):
        raise NotImplementedError("Override the call method for inherited class")
