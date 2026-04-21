import os
import json
from typing import Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
#####################################################################
def save_json(path: str, obj: Any) -> None:
    dir_name = os.path.dirname(path)  #xtracts directory from file path
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
#####################################################################
def pltConfMatrix(cm: np.ndarray, labels: List[str], out_path: str, title: str) -> None:  # reates a visual confusion matrix image and saves it as a png
    plt.figure(figsize=(8, 6))                                   
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)        
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    thresh = cm.max() / 2 if cm.size else 0   

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(int(cm[i, j]), "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",   
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)                                              # Save plot to specified output directory
    plt.close()                                                        # Close figure to free up memory resources
#####################################################################
def evaluate_best(results: Dict[str, Any], out_dir: str) -> Dict[str, Any]: #for evaluating the best model after training

    best = results["best"]                                             # Retrieve the top performing model record
    y_test = best["y_test"]
    y_pred = best["y_pred"]
    
    labels = sorted(set(y_test) | set(y_pred))                # gets all unique labels
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plot_labels = [str(l) for l in labels] 
    
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    
    pltConfMatrix(
        cm, plot_labels,
        cm_path, 
        f"Confusion Matrix: {best['name']}"
    )

    cls_report_text = classification_report(y_test, y_pred, zero_division=0) # Text report
    cls_report_dict = classification_report(y_test, y_pred, output_dict=True) # Dict report
    
    return {
        "best_metrics": best["metrics"],                             
        "all_metrics": results["all_metrics"],                       
        "confusion_matrix_path": cm_path,                           
        "classification_report": cls_report_text,                 
        "per_class_metrics": cls_report_dict,                       
        "confusion_matrix": cm.tolist(),                         
    }
#####################################################################
def write_markdown_report( #generates a full markdown report of the entire ML run
    out_path: str,
    ctx: Any,
    fingerprint: str,
    dataset_profile: Dict[str, Any],
    plan: List[str],
    eval_payload: Dict[str, Any],
    reflection: Dict[str, Any],
) -> None:

    best = eval_payload["best_metrics"]
    numeric = dataset_profile["feature_types"]["numeric"]
    categorical = dataset_profile["feature_types"]["categorical"]
    notes = dataset_profile.get("notes", [])

    def short_list(items: List[str], limit: int = 10) -> str:           # Helper to prevent report cluttering
        if len(items) <= limit: 
            return ", ".join(items)
        return ", ".join(items[:limit]) + f"... (+{len(items)-limit} more)"

    all_models_table = "\n".join([
        f"| {m['model']} | {m.get('balanced_accuracy', 0):.3f} | {m.get('f1_macro', 0):.3f} |"
        for m in eval_payload.get("all_metrics", [])
    ])

    md = f"""# Agentic Data Scientist Report

**Run ID:** `{ctx.run_id}`  
**Started (UTC):** {ctx.started_at}  
**Dataset:** `{ctx.data_path}`  
**Target:** `{ctx.target}`  
**Fingerprint:** `{fingerprint}`  

## Dataset Profile
- Rows: **{dataset_profile["shape"]["rows"]}**
- Columns: **{dataset_profile["shape"]["cols"]}**
- Classification: **{dataset_profile.get("is_classification")}**
- Imbalance ratio: **{dataset_profile.get("imbalance_ratio")}**

**Feature Types**
- Numeric ({len(numeric)}): {short_list(numeric)}
- Categorical ({len(categorical)}): {short_list(categorical)}

**Notes**
{chr(10).join([f"- {n}" for n in notes]) if notes else "- (none)"}

## Plan
{chr(10).join([f"- {t}" for t in plan])}

## All Models Comparison
| Model | Balanced Accuracy | F1 Macro |
| :--- | :--- | :--- |
{all_models_table}

## Results (Best Model: `{best.get("model")}`)
| Metric | Value |
| :--- | :--- |
| Accuracy | {best.get("accuracy", 0):.3f} |
| Balanced Accuracy | {best.get("balanced_accuracy", 0):.3f} |
| Macro F1-Score | {best.get("f1_macro", 0):.3f} |
| Macro Precision | {best.get("precision_macro", 0):.3f} |
| Macro Recall | {best.get("recall_macro", 0):.3f} |

## Confusion Matrix
![Confusion Matrix]({eval_payload.get("confusion_matrix_path")})

## Classification Report
{eval_payload.get("classification_report")}

## Agent Reflection
**Status:** `{reflection.get("status", "N/A")}`  
**Re-plan Recommended:** `{reflection.get("replan_recommended", False)}`

### Identified Issues
{chr(10).join([f"- {i}" for i in reflection.get("issues", [])]) if reflection.get("issues") else "- None"}

### Suggestions
{chr(10).join([f"- {s}" for s in reflection.get("suggestions", [])]) if reflection.get("suggestions") else "- None"}

## Reflection Details
- Improvement vs Dummy (Balanced Acc): **{reflection.get("summary", {}).get("improvement_vs_dummy_balanced_accuracy", 0):.3f}**
- Gap to Second Model (Balanced Acc): **{reflection.get("summary", {}).get("gap_to_second_model_balanced_accuracy", 0):.3f}**
- Gap to Second Model (F1 Macro): **{reflection.get("summary", {}).get("gap_to_second_model_f1_macro", 0):.3f}**

---

*Generated automatically by CE888 Agentic Data Scientist System*
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)                                                    # Write finalized markdown to file