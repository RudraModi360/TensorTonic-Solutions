import math

def evaluate_shadow(production_log, shadow_log, criteria):
    """
    Evaluate whether a shadow model is ready for promotion.
    """
    matching_preds = 0
    prod_correct = 0
    shadow_correct = 0
    shadow_lat = []

    n = min(len(production_log), len(shadow_log))

    for prod, shadow in zip(production_log, shadow_log):
        if prod['actual'] == prod['prediction']:
            prod_correct += 1

        if shadow['actual'] == shadow['prediction']:
            shadow_correct += 1

        if prod['prediction'] == shadow['prediction']:
            matching_preds += 1

        shadow_lat.append(shadow['latency_ms'])

    # Metrics
    prod_acc = prod_correct / n
    shadow_acc = shadow_correct / n
    agreement_rate = matching_preds / n
    acc_gain = shadow_acc - prod_acc

    # P95 latency
    shadow_lat.sort()
    rank = math.ceil(0.95 * n) - 1
    p95 = shadow_lat[rank]

    # Promotion decision
    promote = (
        p95 <= criteria['max_latency_p95'] and
        acc_gain >= criteria['min_accuracy_gain'] and
        agreement_rate >= criteria['min_agreement_rate']
    )

    return {
        "promote": promote,
        "metrics": {
            "shadow_accuracy": shadow_acc,
            "production_accuracy": prod_acc,
            "accuracy_gain": acc_gain,
            "shadow_latency_p95": p95,
            "agreement_rate": agreement_rate
        }
    }