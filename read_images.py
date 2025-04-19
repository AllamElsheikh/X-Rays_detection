import os 
import cv2
import argparse

def read_imgs(dir_files , dim):
  images = []
  labels = []

  fractured_dir = os.path.join(dir_files , 'fractured')
  fractured_files = os.listdir(fractured_dir)
  
  for i , img in enumerate(fractured_files):
    try:
      imrgb = cv2.imread(os.path.join(fractured_dir , img) , 1)
      resized = cv2.resize(imrgb , dim , interpolation=cv2.INTER_AREA) / 255.
      images.append(resized)
      labels.append(1)

    except:
      pass

  not_fractured_dir = os.path.join(dir_files , 'not_fractured')
  not_fractured_files = os.listdir(not_fractured_dir)

  for i , img in enumerate(not_fractured_files):
    try :
      imrgb = cv2.imread(os.path.join(not_fractured_dir , img),  1)
      resized = cv2.resize(imrgb , dim , interpolation=cv2.INTER_AREA) / 255.
      images.append(resized)
      labels.append(0)
    except :
      pass
    return images , labels   



# if __name__ == "__main":
#   parser = argparse.ArgumentParser(description="Read and preprocess X-ray images.")
#   parser.add_argument("--dir" , type=str , required=True ,help='Path to dataset directory')
#   parser.add_argument('--dim', type=int, nargs=2, default=[224, 224],
#                         help='Image dimensions as width height (e.g., --dim 224 224)')
#   args = parser.parse_args()
#   train_ds = read_imgs(args.dir, tuple(args.dim))
#   print(f"✅ Dataset loaded. Total samples: {len(train_ds)}")
#   print(f"🖼️ Sample image shape: {train_ds[0][0].shape}, Label: {train_ds[0][1]}")