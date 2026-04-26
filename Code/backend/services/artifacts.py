import os
import joblib
import numpy as np
import pandas as pd

from backend import state
from backend.config import MODEL_PATH, FEATURES_PATH, TRAINING_X_NPY


def load_artifacts():
    try:
        state.model = joblib.load(MODEL_PATH)
        raw_features = joblib.load(FEATURES_PATH)

        if isinstance(raw_features, (pd.Index, np.ndarray)):
            state.model_features = list(map(str, list(raw_features)))
        elif isinstance(raw_features, list):
            state.model_features = list(map(str, raw_features))
        else:
            state.model_features = [str(x) for x in raw_features]

        print("✅ Model and features loaded successfully")
        print(f"📊 Features: {len(state.model_features)}")
        print(f"📊 Model type: {type(state.model).__name__}")
        return True

    except Exception as e:
        print(f"❌ Failed to load artifacts: {e}")
        state.model = None
        state.model_features = None
        return False


def load_training_data():
    if state.training_data_cache is not None:
        return state.training_data_cache

    if os.path.exists(TRAINING_X_NPY):
        try:
            arr = np.load(TRAINING_X_NPY, allow_pickle=True)
            state.training_data_cache = np.asarray(arr, dtype=float)
            print("✅ Loaded train_X.npy for LIME")
            return state.training_data_cache
        except Exception as e:
            print(f"⚠️ Could not load train_X.npy: {e}")

    return None