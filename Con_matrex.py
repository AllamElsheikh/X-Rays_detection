from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(all_true_labels, all_predicted_classes))

conf_matrix = confusion_matrix(all_true_labels, all_predicted_classes)
print("Confusion Matrix:")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Not fractured', 'Fractured'], yticklabels=['Not Fractured', 'Fractured'])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual Values")

plt.show()