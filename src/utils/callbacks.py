import time
import tensorflow as tf
import os

class TrainingTimeCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - self.epoch_start_time
        self.epoch_times.append(epoch_time)

    def on_train_begin(self, logs=None):
        self.epoch_times = []
        
def cp_cb_generator(cp_file_path):
    # cp_path = os.path.join(cfg.tb_output_dir,cp_filename)
    return tf.keras.callbacks.ModelCheckpoint(   filepath=cp_file_path,
                                                    save_weights_only=True,
                                                    monitor='val_loss',
                                                    mode='min',
                                                    save_best_only=True,
                                                    verbose=1)

if __name__=='__main__':
    pass