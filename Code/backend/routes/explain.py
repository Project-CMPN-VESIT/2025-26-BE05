from flask import Blueprint, jsonify, request

from backend import state
from backend.services.preprocessing import preprocess_data
from backend.services.scoring import predict_score, score_to_risk
from backend.services.explainability import get_shap_explanation, get_lime_explanation
from backend.services.reasoning import generate_rejection_reasons, generate_explanation_summary

bp = Blueprint("explain", __name__)


@bp.route("/explain", methods=["POST", "OPTIONS"])
def explain():
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

        shap_explanation = get_shap_explanation(df_encoded)
        lime_explanation = get_lime_explanation(df_encoded)

        positive_features = []
        negative_features = []
        if shap_explanation and isinstance(shap_explanation.get("all_features"), list):
            positive_features = [
                x for x in shap_explanation["all_features"]
                if isinstance(x, dict) and x.get("shap_value", 0) > 0
            ][:3]
            negative_features = [
                x for x in shap_explanation["all_features"]
                if isinstance(x, dict) and x.get("shap_value", 0) < 0
            ][:5]
        approval_factors = [
            {
                "factor": x["feature"].replace("_", " ").title(),
                "impact": "positive",
                "contribution": abs(x["magnitude"])
            }
            for x in positive_features
        ]

        rejection_reasons = generate_rejection_reasons(negative_features) if approval_status != "approved" else []
        summary = generate_explanation_summary(approval_status, rejection_reasons, approval_factors, score)

        if shap_explanation and "all_features" in shap_explanation:
            inc = len([x for x in shap_explanation["all_features"] if x["shap_value"] > 0])
            dec = len([x for x in shap_explanation["all_features"] if x["shap_value"] < 0])
            total = max(inc + dec, 1)
            impact_distribution = {
                "risk_increasing_pct": round((inc / total) * 100, 2),
                "risk_reducing_pct": round((dec / total) * 100, 2)
            }
        else:
            impact_distribution = {
                "risk_increasing_pct": 0,
                "risk_reducing_pct": 0
            }

        return jsonify({
            "predicted_credit_risk_score": round(score, 2),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "probability_of_default": round(max(0.02, min(0.50, (900 - score) / 1000)), 4),
            "approval_status": approval_status,
            "approval_message": approval_message,
            "total_features_used": len(df_encoded.columns),
            "status": "success",
            "feature_importance_explanation": shap_explanation,
            "lime_explanation": lime_explanation,
            "rejection_reasons": rejection_reasons,
            "approval_factors": approval_factors,
            "impact_distribution": impact_distribution,
            "explanation_summary": summary
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "error"
        }), 500