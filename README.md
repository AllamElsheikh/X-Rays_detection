# X-Ray Detection with Deep Learning and Model Pruning
## Overview
This repository contains a deep learning solution for detecting fractures in X-ray images. The model uses convolutional neural networks (CNN) with pruning techniques to improve performance while reducing the size of the model for more efficient deployment. The project includes image preprocessing, model training, pruning, evaluation, and visualization of training results.

## Project Structure
```bash
/repo
│
├── /data                        # Folder containing the dataset
│   ├── /fractured               # Fractured X-ray images
│   ├── /not_fractured           # Non-fractured X-ray images
│
├── /scripts                     # Scripts for model training and evaluation
│   ├── train_model.py           # Script for model training and pruning
│   ├── evaluate_model.py        # Script for evaluating the model on the test dataset
│   ├── predict_and_collect_labels.py # Script for predicting and collecting results on the test set
│
├── /logs                        # Logs folder for pruning summaries and training logs
│
├── model_for_pruning.h5         # Saved model after pruning and training (optional)
└── README.md                    # This README file

```

## Requirements
To run this project, you will need to install the following Python packages:

- tensorflow
- numpy
- opencv-python
- matplotlib
- scikit-learn
- tensorflow_model_optimization
- tempfile (for logging)
You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Dataset
The dataset consists of X-ray images with two categories:

- fractured: Images showing fractured bones.

- ot_fractured: Images showing non-fractured bones.

The dataset is assumed to be stored in the /data folder, with images organized into subfolders: fractured and not_fractured.

## How to Train the Model
### 1. Preparing the Dataset
Ensure your dataset is structured as follows:
```bash
/data
│
├── /fractured               # X-ray images of fractured bones
├── /not_fractured           # X-ray images of non-fractured bones
```
### 2. Training the Model with Pruning
To train the model with pruning, run the following script:

```bash
python scripts/train_model.py
```
This script will:

- Load and preprocess the data.

- Define a CNN model for X-ray fracture detection.

- Apply pruning to the Dense layers during training.

- Save the pruned model as model_for_pruning.h5.

### 3. Monitor Training Progress
Training progress, including pruning steps, is logged in the /logs directory. Logs will provide insights into pruning performance over time.

How to Evaluate the Model
To evaluate the model on the test set, run:

```bash
python scripts/evaluate_model.py
```
This script will:

- Load the pruned model (model_for_pruning.h5).

- Evaluate the model on the test set.

- Output the test loss and test accuracy.

## How to Make Predictions and Collect Results
To make predictions on the test dataset and collect true labels and predicted classes, run:

```bash
python scripts/predict_and_collect_labels.py
```
This script will:

- Iterate over the test data and make predictions.

- Collect the true labels and predicted classes for further analysis.

- Print the accuracy and classification report.


## Conclusion
This project demonstrates how to use deep learning for X-ray fracture detection. By applying model pruning, we can optimize the model for better performance and reduced size. The provided scripts allow you to train, evaluate, and make predictions using the model.

Feel free to modify the code to suit your dataset and requirements!


## Additional Notes:
If you want to use this for other types of images or tasks, modify the dataset preparation and model architecture as needed.

Always ensure the dataset is preprocessed and augmented as needed to improve model generalization.

You can use the saved model "X-ray_model  ".
