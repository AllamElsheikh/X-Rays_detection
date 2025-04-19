import numpy  as np
from read_images import read_imgs

dim = [180, 180 ]
train_dir = "/content/X-rays_detection/Bone Fracture Dataset/training"
valid_dir = "/content/X-rays_detection/Bone Fracture Dataset/testing"
train_ds = read_imgs(train_dir , dim )
valid_ds = read_imgs(valid_dir  , dim)
images, labels = train_ds
val_images, val_labels = valid_ds
def spliting_data() :
  
  ds_size = len(images)
  train_size = int(.8 * ds_size)

  indices = np.random.permutation(ds_size)


  train_indices = indices[:train_size]
  test_indices = indices[train_size:]


  train_imgs = [images[i] for i in train_indices]
  train_labels = [labels[i] for i in train_indices]

  test_imgs = [images[i] for i in test_indices]
  test_labels = [labels[i] for i in test_indices]
  train_ds = (train_imgs, train_labels)
  test_ds = (test_imgs, test_labels)
  return train_ds , test_ds , valid_ds




















