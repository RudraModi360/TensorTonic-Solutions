def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    total_ref = sum(reference_counts)
    total_prod = sum(production_counts)

    if total_ref == 0 or total_prod == 0:
        return {"score": 0.0, "drift_detected": False}

    ref_probs = [x / total_ref for x in reference_counts]
    prod_probs = [x / total_prod for x in production_counts]

    tvd = 0.0
    for ref, prod in zip(ref_probs, prod_probs):
        tvd += abs(ref - prod)

    tvd *= 0.5

    return {
        "score": tvd,
        "drift_detected": tvd > threshold
    }