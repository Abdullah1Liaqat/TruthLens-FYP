import re

# ===============================
# CREDIT / RISK SIGNALS
# ===============================

POSITIVE_SIGNALS = [
    r"\bReuters\b",
    r"\bBBC\b",
    r"\bDawn\b",
    r"\bAssociated Press\b",
    r"\bofficial data\b",
    r"\bpress conference\b",
    r"\bMinistry\b",
    r"\bconfirmed\b",
    r"\breleased statistics\b",
    r"\baccording to (official|government)\b",
]

NEGATIVE_SIGNALS = [
    r"\bunnamed sources\b",
    r"\bsources close to\b",
    r"\bofficials familiar with the matter\b",
    r"\bcirculating online\b",
    r"\binsiders claim\b",
    r"\bobservers say\b",
    r"\bmay have quietly\b",
    r"\bcould be part of\b",
    r"\bnot been confirmed\b",
    r"\bundisclosed\b",
    r"\bsecret plan\b",
    r"\bsynchronized\b",
    r"\bglobal .* control\b",
    r"\bemergency powers\b",
]


# ===============================
# SCORE CALCULATION
# ===============================

def compute_rule_score(text: str) -> int:
    text_lower = text.lower()

    score = 0

    # Positive evidence
    for pattern in POSITIVE_SIGNALS:
        if re.search(pattern, text_lower):
            score += 1

    # Negative evidence
    for pattern in NEGATIVE_SIGNALS:
        if re.search(pattern, text_lower):
            score -= 1

    return score


# ===============================
# MAIN DECISION ENGINE
# ===============================

def apply_rules(text: str, pred_label: str, confidence: float):
    """
    Returns:
        final_label, final_confidence, rule_score, explanation
    """

    rule_score = compute_rule_score(text)

    explanation = []

    # -------------------------------
    # ZONE A: STRONG CONFIDENCE
    # -------------------------------
    if confidence >= 0.85:
        explanation.append("Strong model confidence → minimal adjustment")

        # only mild penalty for strong negative evidence
        if rule_score <= -4:
            confidence -= 0.10
            explanation.append("High-risk misinformation pattern detected")

        return pred_label, round(confidence, 4), rule_score, explanation


    # -------------------------------
    # ZONE B: MEDIUM CONFIDENCE
    # -------------------------------
    if 0.65 <= confidence < 0.85:

        if rule_score <= -3:
            confidence -= 0.20
            explanation.append("Strong negative credibility signals detected")

            if confidence < 0.55:
                pred_label = "FAKE"
                explanation.append("Confidence dropped below threshold → flipped to FAKE")

        elif rule_score >= 3:
            confidence += 0.08
            explanation.append("Strong credibility signals detected")

        return pred_label, round(confidence, 4), rule_score, explanation


    # -------------------------------
    # ZONE C: BORDERLINE CONFIDENCE
    # -------------------------------
    if 0.50 <= confidence < 0.65:

        if rule_score <= -2:
            pred_label = "FAKE"
            confidence = max(0.50, confidence - 0.10)
            explanation.append("Borderline + weak credibility → classified FAKE")

        elif rule_score >= 2:
            pred_label = "REAL"
            confidence = min(0.80, confidence + 0.10)
            explanation.append("Borderline + strong credibility → upgraded REAL")

        return pred_label, round(confidence, 4), rule_score, explanation


    # fallback
    return pred_label, confidence, rule_score, explanation