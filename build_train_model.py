import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import os
import tempfile
import json  # To save history as a json file
from batch_gen import create_gens
# Placeholder for data generators (replace with actual generators)
train_gen , val_gen , test_gen = create_gens()
# Pruning function and schedule
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
input_shape = (180, 180, 3)
epochs = 6

inputs = keras.Input(shape=input_shape, name='input')

# Data Augmentation layer
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
])(inputs)

# Rescaling layer
rescaling = keras.layers.Rescaling(1./255)(data_augmentation)

# Conv Layer 1
conv_1 = keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv_1')(rescaling)
pool_1 = keras.layers.MaxPooling2D((2, 2), name='pool_1')(conv_1)

# Conv Layer 2
conv_2 = keras.layers.Conv2D(16, (3, 3), activation='relu', name='conv_2')(pool_1)
pool_2 = keras.layers.MaxPooling2D((2, 2), name='pool_2')(conv_2)

# Flatten layer
flat = keras.layers.Flatten(name='flatten')(pool_2)

# Dense layer
dense = keras.layers.Dense(64, activation='relu', name='dense')(flat)

# Dropout layer
dropout = keras.layers.Dropout(0.5, name='dropout')(dense)

# Output layer
outputs = keras.layers.Dense(1, activation='sigmoid', name='output')(dropout)

# Base model
model_1 = keras.Model(inputs=inputs, outputs=outputs)
print(type(model_1))

# Define pruning schedule
pruning_schedule = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                        initial_sparsity=0.5,
                        final_sparsity=0.8,
                        begin_step=0,
                        end_step=np.ceil(len(train_gen) * epochs).astype(np.int32)
    )
}

# Helper function for pruning
def apply_pruning_to_dense(layer):
    if isinstance(layer, keras.layers.Dense):
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer

# Clone model for pruning
model_for_pruning = keras.models.clone_model(
    model_1,
    clone_function=apply_pruning_to_dense,
)

# Compile model for pruning
model_for_pruning.compile(
    optimizer='adam',
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# Display model summary
model_for_pruning.summary()

# Setup callbacks
logdir = tempfile.mkdtemp()
pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()

# Early stopping callback
es_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    verbose=1
)

# Pruning summaries callback (logs pruning stats)
log_callback = tfmot.sparsity.keras.PruningSummaries(log_dir=logdir)

# Train the model with pruning
history = model_for_pruning.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[pruning_callback, log_callback, es_callback]
)

# Save the trained model
model_save_path = './pruned_model'
model_for_pruning.save(model_save_path)
print(f"✅ Model trained and saved to {model_save_path}")

# Save training history to a file
history_dict = history.history
with open('training_history.json', 'w') as f:
    json.dump(history_dict, f)
print("dkf")
