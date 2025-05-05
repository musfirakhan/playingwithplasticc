import os
import numpy as np
import pandas as pd
from constants import LSST_FILTER_MAP, LSST_PB_WAVELENGTHS
from preprocess import (
    __transient_trim,
    fit_2d_gp,
    predict_2d_gp,
    remap_filters,
    robust_scale
)
from utils import train_val_test_split
from scipy import stats
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from pylab import rcParams
import george
from astropy.table import Table, vstack
import scipy.optimize as op
from functools import partial

# Set figure size
rcParams['figure.figsize'] = 12, 8

# Define colors and wavelengths
colours = {
    'lsstu': '#9a0eea',
    'lsstg': '#75bbfd',
    'lsstr': '#76ff7b',
    'lssti': '#fdde6c',
    'lsstz': '#f97306',
    'lssty': '#e50000'
}

pb_wavelengths = {
    "lsstu": 3685., 
    "lsstg": 4802., 
    "lsstr": 6231.,
    "lssti": 7542., 
    "lsstz": 8690., 
    "lssty": 9736.
}

# Define new class mapping
NEW_CLASS_MAPPING = {
    90: "SNIa",
    67: "SNIa",       # Remapped SNIa-91bg to SNIa
    52: "SNIa",       # Remapped SNIax to SNIa
    42: "SNII",
    62: "SNIbc",
    95: "SLSN-I",
    15: "TDE",
    64: "KN",
    88: "AGN",
    92: "RRL",
    65: "M-dwarf",
    16: "EB",
    53: "Mira",
    6: "$\mu$-Lens-Single",
}

def remap_classes(df, new_mapping):
    """Remap target classes according to the new mapping."""
    # First map to the new class names
    df['target_name'] = df['target'].map(new_mapping)
    
    # Then create a numerical encoding for the new classes
    unique_classes = sorted(df['target_name'].unique())
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
    df['encoded_target'] = df['target_name'].map(class_to_index)
    
    return df, class_to_index

def create_dataset(X, y, time_steps=1, step=1):
    """Create dataset with time steps."""
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i : (i + time_steps)].values
        labels = y.iloc[i : (i + time_steps)]
        Xs.append(v)
        
        mode_result = stats.mode(labels)
        if hasattr(mode_result, 'mode'):
            mode_value = mode_result.mode.item()
        else:
            mode_value = mode_result[0][0].item()
        
        ys.append(mode_value)
    
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32).reshape(-1, 1)

def generate_gp_all_objects(object_list, obs_transient, filters, timesteps=100):
    """Generate GP predictions for all objects."""
    adf = pd.DataFrame(columns=['mjd', 'lsstg', 'lssti', 'lsstr', 'lsstu', 'lssty', 'lsstz', 'object_id'])
    inverse_pb_wavelengths = {v: k for k, v in LSST_PB_WAVELENGTHS.items()}
    
    for object_id in object_list:
        df = obs_transient[obs_transient['object_id'] == object_id]
        gp_predict = fit_2d_gp(df, pb_wavelengths=LSST_PB_WAVELENGTHS)
        
        gp_times = np.linspace(min(df['mjd']), max(df['mjd']), timesteps)
        gp_wavelengths = np.vectorize(LSST_PB_WAVELENGTHS.get)(filters)
        
        obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)
        obj_gps['filter'] = obj_gps['filter'].map(inverse_pb_wavelengths)
        
        obj_gps = obj_gps.pivot(index='mjd', columns='filter', values='flux')
        obj_gps = obj_gps.reset_index()
        obj_gps['object_id'] = object_id
        adf = pd.concat([adf, obj_gps])
    
    # Ensure correct data types
    adf['mjd'] = adf['mjd'].astype(np.float32)
    for filter_name in ['lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty']:
        if filter_name in adf.columns:
            adf[filter_name] = adf[filter_name].astype(np.float32)
    adf['object_id'] = adf['object_id'].astype(np.int32)
    
    return adf

def preprocess_data(df, filters):
    """Preprocess the data including class encoding with new mapping."""
    # Remap and encode classes
    df, class_to_index = remap_classes(df, NEW_CLASS_MAPPING)
    
    # Split data
    df_train, df_val, df_test, num_features = train_val_test_split(df, filters)
    
    # Scale features
    robust_scale(df_train, filters)
    robust_scale(df_val, filters)
    robust_scale(df_test, filters)
    
    return df_train, df_val, df_test, num_features, class_to_index

def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, save_dir, class_mapping, class_to_index):
    """Save processed data with class information."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to one-hot encoding
    y_train = to_categorical(y_train, num_classes=len(class_to_index))
    y_val = to_categorical(y_val, num_classes=len(class_to_index))
    y_test = to_categorical(y_test, num_classes=len(class_to_index))
    
    # Create dummy redshift arrays (if needed for training script)
    Z_train = np.zeros((X_train.shape[0], 1), dtype=np.float32)
    Z_test = np.zeros((X_test.shape[0], 1), dtype=np.float32)
    
    # Save numpy arrays
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "X_val.npy"), X_val)
    np.save(os.path.join(save_dir, "y_val.npy"), y_val)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    np.save(os.path.join(save_dir, "Z_train.npy"), Z_train)
    np.save(os.path.join(save_dir, "Z_test.npy"), Z_test)
    
    # Save class mappings
    np.save(os.path.join(save_dir, "class_mapping.npy"), class_mapping)
    np.save(os.path.join(save_dir, "class_to_index.npy"), class_to_index)
    
    # Verification
    print("\nData Verification:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape} (one-hot encoded)")
    print(f"Unique classes: {np.unique(np.argmax(y_train, axis=1))}")
    print(f"Class mapping: {class_mapping}")
    print(f"Class to index: {class_to_index}")
    print(f"\nData saved to {save_dir}")

def main():
    # Create directories
    os.makedirs('data/plasticc/processed', exist_ok=True)
    
    # Load and preprocess data
    print("Loading data...")
    data = pd.read_csv("data/plasticc/training_set.csv", sep=',')
    data = remap_filters(data, {0: 'lsstu', 1: 'lsstg', 2: 'lsstr', 
                               3: 'lssti', 4: 'lsstz', 5: 'lssty'})
    data.rename({'flux_err': 'flux_error'}, axis='columns', inplace=True)
    
    # Get unique filters and objects
    filters = list(np.unique(data['filter']))
    object_list = list(np.unique(data['object_id']))
    
    # Trim to transient period
    print("Trimming transients...")
    obs_transient, object_list = __transient_trim(object_list, data)
    
    # Generate GP predictions
    print("Generating GP predictions...")
    generated_gp_dataset = generate_gp_all_objects(object_list, obs_transient, filters)
    
    # Load and merge metadata
    print("Merging metadata...")
    metadata_pd = pd.read_csv("data/plasticc/training_set_metadata.csv", sep=',', index_col='object_id')
    metadata_pd = metadata_pd.reset_index()
    metadata_pd['object_id'] = metadata_pd['object_id'].astype(np.int32)
    metadata_pd['target'] = metadata_pd['target'].astype(np.int32)
    
    # Merge and clean data
    df_combi = generated_gp_dataset.merge(metadata_pd, on='object_id', how='left')
    df = df_combi.drop(columns=[
        "ra", "decl", "gal_l", "gal_b", "ddf",
        "hostgal_specz", "hostgal_photoz", "hostgal_photoz_err",
        "distmod", "mwebv",
    ])
    
    # Split and scale data with new class mapping
    print("Preprocessing data...")
    df_train, df_val, df_test, num_features, class_to_index = preprocess_data(df, filters)
    
    # Create datasets
    TIME_STEPS = 20
    STEP = 20
    
    X_train, y_train = create_dataset(
        df_train[filters],
        df_train.encoded_target,
        TIME_STEPS,
        STEP
    )
    
    X_val, y_val = create_dataset(
        df_val[filters],
        df_val.encoded_target,
        TIME_STEPS,
        STEP
    )
    
    X_test, y_test = create_dataset(
        df_test[filters],
        df_test.encoded_target,
        TIME_STEPS,
        STEP
    )
    
    # Save processed data
    print("Saving processed data...")
    save_processed_data(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        "data/plasticc/processed/",
        NEW_CLASS_MAPPING,
        class_to_index
    )
    
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()