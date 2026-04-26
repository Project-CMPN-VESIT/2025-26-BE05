import os
from flask import Blueprint, jsonify

from backend import state
from backend.config import TRAINING_X_NPY
from backend.services.explainability import SHAP_AVAILABLE, LIME_AVAILABLE

bp = Blueprint("main", __name__)


@bp.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "Credit Risk Prediction API with Explainability is running",
        "model_loaded": state.model is not None,
        "features_count": len(state.model_features) if state.model_features else 0,
        "explainability_methods": ["SHAP", "LIME"],
        "shap_available": SHAP_AVAILABLE,
        "lime_available": LIME_AVAILABLE,
        "endpoints": {
            "/predict": "Prediction only",
            "/explain": "Prediction + SHAP + LIME"
        }
    })


@bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": state.model is not None,
        "features_loaded": state.model_features is not None,
        "training_data_loaded": os.path.exists(TRAINING_X_NPY),
        "shap_available": SHAP_AVAILABLE,
        "lime_available": LIME_AVAILABLE
    })