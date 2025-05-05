import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from astronet.t2.model import T2Model
from astronet.utils import find_optimal_batch_size
from astronet.metrics import WeightedLogLoss, DistributedWeightedLogLoss
from astronet.viz.visualise_results import plot_confusion_matrix

# Load test data
X_test = np.load('data/plasticc/processed/X_test.npy')
y_test = np.load('data/plasticc/processed/y_test.npy')
Z_test = np.load('data/plasticc/processed/Z_test.npy')

# Load model from latest checkpoint with custom loss function
model = tf.keras.models.load_model(
    'astronet/t2/models/plasticc/checkpoints/checkpoint-1746297954-None-0.10.0',
    custom_objects={
        'WeightedLogLoss': WeightedLogLoss,
        'DistributedWeightedLogLoss': DistributedWeightedLogLoss
    }
)

# Make predictions
y_pred = model.predict({'input_1': X_test, 'input_2': Z_test})
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Create confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Plot confusion matrix using matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes))

# Calculate and print accuracy
accuracy = np.mean(y_pred_classes == y_true_classes)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Plot confusion matrix using astronet.visualise
plot_confusion_matrix(
    't2',
    'plasticc',
    'checkpoint-1746297954-None-0.10.0',
    y_test,
    y_pred,
    'None',
    class_names=None,
    cmap=None,
    save=True,
) 