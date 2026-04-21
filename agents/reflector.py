from typing import Any, Dict, List, Tuple

#####################################################################
def safeFloat(value: Any, default: float = 0.0) -> float: #defensive utility function
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

#####################################################################
def reflect(                                                       #takes model results, dataset info and decides if the result is good or bad. also gives feedback
    dataset_profile: Dict[str, Any],
    evaluation: Dict[str, Any],
    all_metrics: List[Dict[str, Any]]
) -> Dict[str, Any]:

    ballACC = safeFloat(evaluation.get("balanced_accuracy"))  # Extract main evaluation metrics safely

    f1 = safeFloat(evaluation.get("f1_macro"))

    precision = safeFloat(evaluation.get("precision_macro"))
    recall = safeFloat(evaluation.get("recall_macro"))

    imb = safeFloat(dataset_profile.get("imbalance_ratio", 1.0))
    rows = int(dataset_profile.get("shape", {}).get("rows", 0))

    issues: List[str] = []             # Stores the detected problems
    suggestions: List[str] = []        # improvement ideas

    if f1 < 0.5 or ballACC < 0.5:              #bad model
        status = "poor"

    elif f1 < 0.7:                             #average model
        status = "average"

    else:                                      #nice model
        status = "good"

    if f1 < 0.6:
        issues.append("Low overall F1")        #weak performance

    if ballACC < 0.55:                         #not balanced
        issues.append("Low balanced accuracy")

    if abs(precision - recall) > 0.15:
        issues.append("Model bias (precision vs recall imbalance)")

        if precision > recall:
            suggestions.append("Model is conservative. improve recall with class_weight or different model.")
        else:
            suggestions.append("Model is over-predicting. improve precision with stronger regularization or different model.")

    if imb >= 5:
        issues.append("Severe class imbalance")

        if recall < 0.5:
            suggestions.append("Minority class likely ignored. use class_weight or resampling.")

    if rows < 100:
        issues.append("Very small dataset")
        suggestions.append("High variance risk. use cross-validation or simpler models.")

    dummy = next((m for m in all_metrics if "Dummy" in m.get("model", "")), None)

    improvement = 0.0
    
    if dummy:
        dummy_score = safeFloat(dummy.get("balanced_accuracy"))
        improvement = ballACC - dummy_score

        if improvement < 0.05:
            issues.append("Model barely better than baseline")
            suggestions.append("Feature signal is weak; try different models or better preprocessing.")

    if len(all_metrics) > 1:
        scores = [safeFloat(m.get("balanced_accuracy")) for m in all_metrics]
        if max(scores) - min(scores) > 0.3:
            issues.append("High variance across models")
            suggestions.append("Model selection unstable; prefer more robust models.")

    replan = False

    if status == "poor":
        replan = True

    if "Severe class imbalance" in issues:
        replan = True

    if "Model barely better than baseline" in issues:
        replan = True

    return {
        "status": status,
        "issues": issues,
        "suggestions": list(set(suggestions)),
        "replan_recommended": replan,
        "summary": {
            "balanced_accuracy": ballACC,
            "f1_macro": f1,
            "precision_macro": precision,
            "recall_macro": recall,
            "improvement_vs_dummy_balanced_accuracy": improvement,
        },
    }
#####################################################################
def should_replan(reflection: Dict[str, Any]) -> bool:
    return bool(reflection.get("replan_recommended", False))
#####################################################################
def apply_replan_strategy(    #for modifying plan based on reflection

    plan: List[str],
    dataset_profile: Dict[str, Any],
    reflection: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:

    newPlan = list(plan)
    newProfile = dict(dataset_profile)

    if "replan_attempt" not in newPlan:
        newPlan.append("replan_attempt")

    issues = reflection.get("issues", [])

    if "Low overall F1" in issues:
        if "try_more_models" not in newPlan:
            newPlan.insert(newPlan.index("select_models"), "try_more_models")

    if "Severe class imbalance" in issues:
        if "handle_imbalance" not in newPlan:
            newPlan.insert(newPlan.index("train_models"), "handle_imbalance")

    if "Very small dataset" in issues:
        if "use_cross_validation" not in newPlan:
            newPlan.insert(newPlan.index("train_models"), "use_cross_validation")

    deduped = []
    for step in newPlan:
        if step not in deduped:
            deduped.append(step)

    return deduped, newProfile