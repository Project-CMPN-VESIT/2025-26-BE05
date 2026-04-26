import pandas as pd
from backend import state
from backend.constants import CATEGORICAL_COLS, FLAG_COLS, NUMERIC_COLS


def normalize_bool(value):
    if isinstance(value, str):
        return 1 if value.strip().lower() in ("true", "1", "yes", "y") else 0
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return int(value)
    except Exception:
        return 0


def preprocess_data(data):
    df = pd.DataFrame([data])

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            df[col] = "Unknown"
        else:
            df[col] = df[col].astype(str).fillna("Unknown")

    for col in FLAG_COLS:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].apply(normalize_bool)

    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    if state.model_features is None:
        raise ValueError("model_features not loaded")

    for col in state.model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    extra_cols = [c for c in df_encoded.columns if c not in state.model_features]
    if extra_cols:
        df_encoded = df_encoded.drop(columns=extra_cols)

    df_encoded = df_encoded[state.model_features]
    df_encoded = df_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df_encoded