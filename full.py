# !pip install gpflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import gpflow
from gpflow.utilities import positive

def safe_gp_interpolation(data, object_ids, time_points=100):
    interpolated_data = []

    for obj_id in object_ids:
        obj_data = data[data['object_id'] == obj_id]

        global_min_time = data['mjd'].min()
        global_max_time = data['mjd'].max()
        pred_times = np.linspace(global_min_time, global_max_time, time_points)

        for pb in range(6):  # 6 passbands
            pb_data = obj_data[obj_data['passband'] == pb]

            if len(pb_data) < 3:
                for t in pred_times:
                    interpolated_data.append([obj_id, t, pb, 0.0])
                continue

            times = pb_data['mjd'].values.reshape(-1, 1)
            fluxes = pb_data['flux'].values.reshape(-1, 1)
            flux_errs = pb_data['flux_err'].values

            flux_errs = np.where(flux_errs <= 0, np.median(flux_errs[flux_errs > 0]) if np.any(flux_errs > 0) else 1.0, flux_errs)

            try:
                kernel = gpflow.kernels.Matern32()
                model = gpflow.models.GPR(
                    data=(times, fluxes),
                    kernel=kernel,
                    noise_variance=np.median(flux_errs ** 2)  # FIXED: use scalar value
                )

                mean, _ = model.predict_f(pred_times.reshape(-1, 1))
                fluxes_interp = mean.numpy().flatten()

            except Exception as e:
                from scipy.interpolate import interp1d
                interp_func = interp1d(times.flatten(), fluxes.flatten(), bounds_error=False, fill_value=0.0)
                fluxes_interp = interp_func(pred_times)

            for t, flux in zip(pred_times, fluxes_interp):
                interpolated_data.append([obj_id, t, pb, flux])

    return pd.DataFrame(interpolated_data, columns=['object_id', 'mjd', 'passband', 'flux'])
# Load and merge data
train_meta = pd.read_csv('ML-project\\Data\\Plasticc\\Training')
train_lc = pd.read_csv('ML-project\\Data\\Plasticc\\training_set.csv')
merged_data = train_lc.merge(train_meta, on='object_id')

# Split data
train_data, test_data = train_test_split(merged_data, test_size=0.15,
                                       random_state=42,
                                       stratify=merged_data['target'])

# Get unique object IDs
train_object_ids = train_data['object_id'].unique()
test_object_ids = test_data['object_id'].unique()

# Interpolate with the safe version
print("Interpolating training data...")
train_interp = safe_gp_interpolation(train_data, train_object_ids)

print("Interpolating test data...")
test_interp = safe_gp_interpolation(test_data, test_object_ids)

# Reshape data function
def reshape_interpolated_data(interp_df, meta_df, time_points=100):
    # Pivot to wide format
    wide_df = interp_df.pivot_table(
        index='object_id',
        columns=['mjd', 'passband'],
        values='flux',
        fill_value=0.0  # Fill missing with 0
    )

    # Ensure consistent columns
    cols = sorted(wide_df.columns)
    wide_df = wide_df[cols]

    # Reshape to (samples, time_steps, features)
    X = wide_df.values.reshape(len(wide_df), time_points, 6)

    # Get metadata in same order
    meta = meta_df.set_index('object_id').loc[wide_df.index]
    features = meta[['hostgal_photoz', 'hostgal_photoz_err', 'mwebv']].values
    labels = meta['target'].values

    return X, features, labels

# Prepare final datasets
X_train, feat_train, y_train = reshape_interpolated_data(train_interp, train_meta)
X_test, feat_test, y_test = reshape_interpolated_data(test_interp, train_meta)

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np

# Step 1: Use original labels (e.g., from `meta['target']`)
y_train_raw = train_meta.set_index('object_id').loc[train_interp['object_id'].unique()]['target'].values
y_test_raw = train_meta.set_index('object_id').loc[test_interp['object_id'].unique()]['target'].values

# Step 2: Label encode
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_raw)
y_test_encoded = label_encoder.transform(y_test_raw)

# Step 3: Get the correct number of classes
num_classes = len(np.unique(y_train_encoded))

# Step 4: One-hot encode
y_train = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)

def build_t2_model(num_classes=14, d_model=32, num_heads=16, dff=128, dropout_rate=0.1):
    # Input layers
    time_series_input = tf.keras.Input(shape=(100, 6), name='time_series')  # 100 time points, 6 passbands
    additional_features = tf.keras.Input(shape=(3,), name='additional_features')  # redshift, redshift_err, MWEBV

    # Convolutional embedding (as in paper)
    x = layers.Conv1D(d_model, kernel_size=1, activation='relu')(time_series_input)

    # Positional encoding
    position = tf.range(start=0, limit=100, delta=1, dtype=tf.float32)
    angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model)//2) / np.float32(d_model)))
    angle_rads = position[:, np.newaxis] * angle_rates[np.newaxis, :]

    # Apply sin to even indices and cos to odd indices
    angle_rads_even = tf.math.sin(angle_rads[:, 0::2])
    angle_rads_odd = tf.math.cos(angle_rads[:, 1::2])
    angle_rads = tf.concat([angle_rads_even, angle_rads_odd], axis=-1)
    positional_encoding = angle_rads[np.newaxis, ...]

    # Add positional encoding to time series
    x += positional_encoding

    # Multi-head attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model//num_heads
    )(x, x)

    # Skip connection and layer norm
    x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

    # Feed forward network
    ffn = tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ])
    ffn_output = ffn(x)

    # Skip connection and layer norm
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # Global average pooling (as in paper)
    x = layers.GlobalAveragePooling1D()(x)

    # Concatenate additional features
    x = layers.concatenate([x, additional_features])

    # Final classification layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=[time_series_input, additional_features], outputs=outputs)

test_meta = train_meta[train_meta['object_id'].isin(test_object_ids)]

# Build model (same as before)
model = build_t2_model()

# Compile with custom learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.017)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
]

# Train
history = model.fit(
    [X_train, feat_train],
    y_train,
    validation_data=([X_test, feat_test], y_test),
    batch_size=32,
    epochs=130,
    callbacks=callbacks,
    verbose=1
)
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer

def calculate_weighted_log_loss(y_true, y_pred, class_weights=None):
    """
    Calculate weighted log loss as per PLAsTiCC metric

    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        class_weights: Dictionary of class weights (if None, uses flat weights)

    Returns:
        weighted_log_loss: Calculated weighted log loss
    """
    if class_weights is None:
        # Use flat weights as in the paper if not specified
        class_weights = {i: 1.0 for i in range(y_pred.shape[1])}

    # Get number of samples per class
    class_counts = np.sum(y_true, axis=0)

    # Calculate per-class log loss
    per_class_loss = []
    for j in range(y_pred.shape[1]):
        class_mask = y_true[:, j] == 1
        if np.sum(class_mask) > 0:  # Only calculate if class exists in y_true
            class_loss = log_loss(y_true[class_mask][:, j],
                                y_pred[class_mask][:, j],
                                labels=np.arange(y_pred.shape[1]))
            per_class_loss.append(class_weights[j] * class_loss)

    # Weighted average
    weighted_log_loss = np.sum(per_class_loss) / np.sum(list(class_weights.values()))
    return weighted_log_loss

def test_model_and_save_results(model, test_data, test_meta, output_file='test_results.csv'):
    """
    Test the model and save results in required format

    Args:
        model: Trained t2 model
        test_data: Test set light curve data
        test_meta: Test set metadata
        output_file: Path to save results CSV
    """
    # Get unique object IDs
    test_object_ids = test_data['object_id'].unique()

    # Interpolate test data
    print("Interpolating test data...")
    test_interp = safe_gp_interpolation(test_data, test_object_ids)

    # Reshape test data
    X_test, feat_test, y_test = reshape_interpolated_data(test_interp, test_meta)

    # Get true labels (for metrics)
    y_test_true = test_meta.set_index('object_id').loc[test_object_ids]['target'].values
    y_test_onehot = tf.keras.utils.to_categorical(y_test_true, num_classes=15)

    # Predict probabilities
    print("Making predictions...")
    y_pred = model.predict([X_test, feat_test])

    # Create results DataFrame
    results = pd.DataFrame(y_pred, columns=[f'P(class_{i})' for i in range(15)])
    results.insert(0, 'Object ID', test_object_ids)

    # Save to CSV
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Calculate metrics
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y_test_true)
    weighted_log_loss = calculate_weighted_log_loss(y_test_onehot, y_pred)

    print("\nTest Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted Log Loss: {weighted_log_loss:.4f}")

    return results, accuracy, weighted_log_loss

# Example usage:
if __name__ == "__main__":
    # Or use the one we built earlier
    model = build_t2_model()
    # Run testing and save results
    results, accuracy, wll = test_model_and_save_results(
        model,
        test_data,
        test_meta,
        output_file='phase1_test_results.csv'
    )

    # Print first few rows
    print("\nSample results:")
    print(results.head())