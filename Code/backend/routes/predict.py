from flask import Blueprint, jsonify, request

from backend import state
from backend.services.preprocessing import preprocess_data
from backend.services.scoring import predict_score, score_to_risk

bp = Blueprint("predict", __name__)


@bp.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    try:
        if state.model is None or state.model_features is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No data provided"}), 400

        df_encoded = preprocess_data(data)
        score = predict_score(df_encoded)
        risk_level, risk_color, approval_status, approval_message = score_to_risk(score)

        return jsonify({
            "predicted_credit_risk_score": round(score, 2),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "probability_of_default": round(max(0.02, min(0.50, (900 - score) / 1000)), 4),
            "approval_status": approval_status,
            "approval_message": approval_message,
            "total_features_used": len(df_encoded.columns),
            "status": "success"
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "error"
        }), 500