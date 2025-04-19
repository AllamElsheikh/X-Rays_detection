import tensorflow as tf
from tensorflow import keras

# Assuming your test dataset is stored in 'test_gen'
# Replace 'test_gen' with your actual test dataset if necessary

# Load the model for pruning (assuming it's saved as 'model_for_pruning.h5' or similar)
# If you don't save the model, you can re-train it in the previous script and then load it here
model_for_pruning = keras.models.load_model('model_for_pruning.h5')  # Adjust the path accordingly

# Evaluate the model
test_loss, test_accuracy = model_for_pruning.evaluate(test_gen, verbose=1)

# Print the results
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
