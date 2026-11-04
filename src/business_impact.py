
from sklearn.metrics import confusion_matrix


def calculate_business_impact(
    y_actual,
    y_predicted,
    cost_per_missed_churner=100,
    cost_per_false_alarm=10
):
    """
    Estimate illustrative business costs.

    Costs are assumptions, not actual company financial data.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_actual,
        y_predicted,
        labels=[0, 1]
    ).ravel()

    missed_churn_cost = (
        fn * cost_per_missed_churner
    )

    false_alarm_cost = (
        fp * cost_per_false_alarm
    )

    total_cost = (
        missed_churn_cost + false_alarm_cost
    )

    return {
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "missed_churn_cost": missed_churn_cost,
        "false_alarm_cost": false_alarm_cost,
        "estimated_total_cost": total_cost
    }
