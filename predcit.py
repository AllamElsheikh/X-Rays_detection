import numpy as np
import tensorflow as tf
from tensorflow import keras

# Assuming your test dataset is stored in 'test_gen'
# Replace 'test_gen' with your actual test dataset if necessary

# Load the trained model
model_for_pruning = keras.models.load_model('model_for_pruning.h5')  # Adjust the path accordingly

# Initialize lists to store all true labels and predicted classes
all_true_labels = []
all_predicted_classes = []

# Loop through the test dataset and collect predictions
for i in range(len(test_gen)):
    # Get a batch of images and labels from the test dataset generator
    images, labels = test_gen[i]

    # Make predictions
    predictions = model_for_pruning.predict(images)
    
    # Convert predictions to binary classes (0 or 1)
    predicted_classes = (predictions > 0.5).astype(int)
    
    # Append the true labels and predicted classes to the respective lists
    all_true_labels.extend(labels)
    all_predicted_classes.extend(predicted_classes)

# Convert the lists to numpy arrays for further analysis
all_true_labels = np.array(all_true_labels)
all_predicted_classes = np.array(all_predicted_classes)

# Optionally, you can print or save the results, for example:
print(f"True Labels: {all_true_labels}")
print(f"Predicted Classes: {all_predicted_classes}")

# If needed, you can compute additional metrics like accuracy, precision, recall, F1-score, etc.
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(all_true_labels, all_predicted_classes)
report = classification_report(all_true_labels, all_predicted_classes)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
