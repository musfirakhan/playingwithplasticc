import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from astronet.viz.visualise_results import (
    plot_acc_history,
    plot_confusion_matrix,
    plot_loss_history,
    plot_multiROC
)
from astronet.metrics import WeightedLogLoss

# Load your saved model and data
model = tf.keras.models.load_model('astronet/t2/models/plasticc/model-myfirstrun')
X_test = np.load('data/plasticc/processed/X_test.npy')
y_test = np.load('data/plasticc/processed/y_test.npy')
Z_test = np.load('data/plasticc/processed/Z_test.npy')

# Make predictions
y_pred = model.predict([X_test, Z_test])

# Get true labels (convert from one-hot)
y_true = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
plot_confusion_matrix(y_true, y_pred_labels)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot ROC curves
plt.figure(figsize=(10, 8))
plot_multiROC(y_test, y_pred)
plt.savefig('roc_curves.png')
plt.close()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_labels))

# Calculate and print weighted log loss
wll = WeightedLogLoss()
log_loss = wll(y_test, y_pred).numpy()
print(f"\nWeighted Log Loss: {log_loss:.4f}")

# If you have training history saved, you can also plot:
# plot_acc_history(history)
# plot_loss_history(history)

model.save('astronet/t2/models/plasticc/model-my_first_run') 