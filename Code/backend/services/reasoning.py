def generate_rejection_reasons(negative_features):
    reasons = []

    for feature in negative_features[:5]:
        name = feature["feature"]
        clean = name.replace("_", " ").title()

        if "Credit_Score" in name:
            reasons.append({
                "factor": "Credit Score",
                "issue": "Low credit score is hurting approval chances",
                "improvement": "Make timely payments and reduce utilization"
            })
        elif "Debt_to_Income_Ratio" in name:
            reasons.append({
                "factor": "Debt-to-Income Ratio",
                "issue": "Debt is high relative to income",
                "improvement": "Reduce debt or increase income"
            })
        elif "Late_Payments" in name:
            reasons.append({
                "factor": "Payment History",
                "issue": "Late payment history is negatively impacting the score",
                "improvement": "Keep all payments on time"
            })
        elif "Credit_Utilization" in name:
            reasons.append({
                "factor": "Credit Utilization",
                "issue": "Credit utilization is too high",
                "improvement": "Try to keep utilization below 30%"
            })
        elif "Bankruptcy" in name:
            reasons.append({
                "factor": "Bankruptcy History",
                "issue": "Bankruptcy history is reducing creditworthiness",
                "improvement": "Build positive credit history over time"
            })
        elif "Annual_Income" in name or "Income" in name:
            reasons.append({
                "factor": "Income Level",
                "issue": "Income may not support the requested loan",
                "improvement": "Increase income or request a smaller loan"
            })
        else:
            reasons.append({
                "factor": clean,
                "issue": f"{clean} is negatively affecting the prediction",
                "improvement": f"Improve {clean.lower()} if possible"
            })

    return reasons


def generate_explanation_summary(status, rejection_reasons, approval_factors, score):
    if status == "approved":
        summary = f"Your application has been approved with a credit risk score of {score:.2f}. "
        if approval_factors:
            top = ", ".join([x["factor"] for x in approval_factors[:3]])
            summary += f"Strong factors include {top}."
        else:
            summary += "Your profile looks favorable."
        return summary

    if status == "conditional":
        summary = f"Your application scored {score:.2f} and needs additional review. "
        if rejection_reasons:
            summary += f"The main concern is {rejection_reasons[0]['factor']}. "
            summary += rejection_reasons[0]["improvement"]
        return summary

    summary = f"Unfortunately, your application was declined with a score of {score:.2f}. "
    if rejection_reasons:
        top = ", ".join([x["factor"] for x in rejection_reasons[:3]])
        summary += f"Key issues: {top}. "
        summary += rejection_reasons[0]["improvement"]
    return summary