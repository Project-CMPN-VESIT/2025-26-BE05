import numpy as np
from backend import state


def score_to_risk(score):
    if score >= 760:
        return "Very Low Risk", "green", "approved", "Approved - Excellent profile"
    elif score >= 660:
        return "Low Risk", "lightgreen", "approved", "Approved - Good profile"
    elif score >= 540:
        return "Medium Risk", "yellow", "conditional", "Conditional approval - review required"
    elif score >= 420:
        return "High Risk", "orange", "rejected", "Rejected - High risk"
    else:
        return "Very High Risk", "red", "rejected", "Rejected - Very high risk"


def predict_score(df_encoded):
    raw = state.model.predict(df_encoded)
    return float(np.clip(raw[0], 300, 900))