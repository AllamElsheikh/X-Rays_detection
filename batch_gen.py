import tensorflow as tf
import numpy as np
from splitting_data import spliting_data
class BatchGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, data, labels, batch_size = 32, dim = (180, 180, 3), shuffle = True):
        self.data = data
        self.labels = labels
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
            
    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        data_temp = [list(self.data)[k] for k in indexes]
        labels_temp = [list(self.labels)[k] for k in indexes]
        X, y = np.asarray(data_temp), np.asarray(labels_temp)
        
        return [X, y]
def create_gens():
  train_ds , test_ds , valid_ds = spliting_data()
  train_gen = BatchGenerator(train_ds[0], train_ds[1])
  val_gen = BatchGenerator(valid_ds[0], valid_ds[1])
  test_gen = BatchGenerator(test_ds[0], test_ds[1])
  return train_gen , val_gen , test_gen

