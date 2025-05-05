import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from astronet.metrics import WeightedLogLoss

# Load test data
X_test = np.load('data/plasticc/processed/X_test.npy')
Z_test = np.load('data/plasticc/processed/Z_test.npy')
y_test = np.load('data/plasticc/processed/y_test.npy')

print("\nOriginal data shapes:")
print(f"X_test shape: {X_test.shape}")
print(f"Z_test shape: {Z_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Make sure all arrays have the same number of samples
n_samples = y_test.shape[0]  # Use y_test as reference
X_test = X_test[:n_samples]
Z_test = Z_test[:n_samples]

print("\nAdjusted data shapes:")
print(f"X_test shape: {X_test.shape}")
print(f"Z_test shape: {Z_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Load the model with custom objects
custom_objects = {'WeightedLogLoss': WeightedLogLoss}
model = tf.keras.models.load_model('astronet/t2/models/plasticc/temp_model', custom_objects=custom_objects)

# Print model information
print("\nModel Configuration:")
model.summary()

# Make predictions
print("\nMaking predictions...")
y_pred = model.predict(X_test, verbose=1)  # Try with just X_test first

# Calculate weighted log loss
print("\nCalculating metrics...")
wloss = WeightedLogLoss()
loss_value = wloss(y_test, y_pred).numpy()
print(f"\nWeighted Log Loss: {loss_value:.3f}")

# Convert predictions to class indices
y_true = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Create confusion matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_true, y_pred_classes)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
plt.title(f'Confusion Matrix (Normalized)\nWeighted Log Loss = {loss_value:.3f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

print("\nConfusion Matrix saved as 'confusion_matrix.png'") 