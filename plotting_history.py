import json
import matplotlib.pyplot as plt

# Load training history from file
with open('training_history.json', 'r') as f:
    history_dict = json.load(f)

# Extract loss and validation loss from the history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

# Plotting the training and validation loss
plt.style.use('ggplot')

plt.figure(figsize=(12, 6))
plt.plot(epochs, loss, 'o-', color='royalblue', label='Training Loss', linewidth=2, markersize=8)
plt.plot(epochs, val_loss, 's--', color='darkorange', label='Validation Loss', linewidth=2, markersize=8)

plt.title('Training and Validation Loss Over Epochs', fontsize=16, fontweight='bold')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)

plt.legend(fontsize=12)

plt.grid(True, alpha=0.3)

plt.xlim([1, len(epochs)])

plt.tight_layout()
plt.savefig('training_loss_plot.png', format='png') 
plt.show()
