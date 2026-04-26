import numpy as np

from backend import state
from backend.services.artifacts import load_training_data

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False


def get_shap_explanation(df_encoded):
    if not SHAP_AVAILABLE or state.model is None:
        return None

    try:
        if state.shap_explainer_cache is None:
            state.shap_explainer_cache = shap.TreeExplainer(state.model)

        explainer = state.shap_explainer_cache
        sample = df_encoded.values.astype(float)
        shap_values = explainer.shap_values(sample)
        expected_value = explainer.expected_value

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        row_vals = shap_values[0]
        feature_values = df_encoded.iloc[0].values.astype(float)

        contributions = []
        for i, fname in enumerate(state.model_features):
            v = float(row_vals[i])
            contributions.append({
                "feature": fname,
                "shap_value": round(v, 6),
                "impact": "positive" if v > 0 else "negative" if v < 0 else "neutral",
                "magnitude": abs(round(v, 6)),
                "actual_value": round(float(feature_values[i]), 6)
            })

        contributions.sort(key=lambda x: x["magnitude"], reverse=True)
        top_positive = [x for x in contributions if x["shap_value"] > 0][:5]
        top_negative = [x for x in contributions if x["shap_value"] < 0][:5]

        return {
            "method": "SHAP",
            "base_value": float(np.squeeze(expected_value)) if np.size(expected_value) == 1 else float(np.ravel(expected_value)[0]),
            "all_features": contributions,
            "top_features": contributions[:10],
            "positive_features": top_positive,
            "negative_features": top_negative,
            "impact_distribution": {
                "risk_increasing_pct": round((len(top_positive) / max(len(top_positive) + len(top_negative), 1)) * 100, 2),
                "risk_reducing_pct": round((len(top_negative) / max(len(top_positive) + len(top_negative), 1)) * 100, 2)
            }
        }

    except Exception as e:
        return {
            "method": "SHAP",
            "error": str(e),
            "all_features": [],
            "top_features": []
        }


def get_lime_explanation(df_encoded):
    if not LIME_AVAILABLE or state.model is None:
        return None

    try:
        data = load_training_data()
        if data is None:
            base = np.tile(df_encoded.values[0], (500, 1))
            noise = np.random.normal(0.0, 1e-3, size=base.shape)
            data = base + noise

        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.asarray(data, dtype=float),
            feature_names=state.model_features,
            mode="regression",
            discretize_continuous=True,
            verbose=False,
            random_state=42
        )

        exp = lime_explainer.explain_instance(
            data_row=df_encoded.values[0].astype(float),
            predict_fn=state.model.predict,
            num_features=10,
            num_samples=1000
        )

        lime_score = None
        try:
            if hasattr(exp, "score"):
                lime_score = float(exp.score)
            elif hasattr(exp, "local_pred"):
                lime_score = float(np.squeeze(exp.local_pred))
        except Exception:
            lime_score = None

        lime_contributions = []
        for feature_str, weight in exp.as_list():
            raw = feature_str.strip()
            feature_name = raw
            for sep in [" <= ", "<=", " >= ", ">=", " = ", "=", " > ", ">", " < ", "<"]:
                if sep in raw:
                    feature_name = raw.split(sep)[0].strip()
                    break

            lime_contributions.append({
                "feature": feature_name,
                "weight": round(float(weight), 6),
                "impact": "positive" if float(weight) > 0 else "negative" if float(weight) < 0 else "neutral",
                "magnitude": abs(round(float(weight), 6)),
                "description": feature_str
            })

        return {
            "method": "LIME",
            "score": lime_score,
            "contributions": lime_contributions
        }

    except Exception as e:
        return {
            "method": "LIME",
            "error": str(e),
            "contributions": []
        }