import numpy as np
from typing import Any, Dict, Optional
import pandas as pd
import hashlib
#####################################################################
def infer_target_column(df: pd.DataFrame) -> Optional[str]: #guesses target column

    if df.shape[1] == 0:
        return None

    n_rows = len(df)
    best_col = None
    bestScore = -float("inf") # start with very low score

    for col in df.columns:

        series = df[col]
        non_null = series.dropna()

        if len(non_null) == 0:
            continue

        uniq = series.nunique(dropna=True)
        ratio = uniq / n_rows if n_rows > 0 else 1.0

        print(f"[DEBUG] col={col} | uniq={uniq} | ratio={ratio:.6f}")

        score = 0.0

        if uniq > 50:
            score -= 5 #penalizes too many unique value

        is_clf = is_classification_target(series)

        if is_clf:
            score += 10 #strong reward for classification-like column
        else:
            score += 2 #small reward for regression-like

        if is_clf:
            if 2 <= uniq <= 5:
                score += 4

            elif 6 <= uniq <= 15:
                score += 6

            elif 16 <= uniq <= 30:
                score += 8

            else:
                score -= 4 # Too many classes 

        if is_clf:
            vc = non_null.value_counts()

            if len(vc) > 1:
                imbalance = vc.max() / max(vc.min(), 1)

                if imbalance < 10:
                    score += 3 #balanced

                else:
                    score -= 2 # imbalanced

        if pd.api.types.is_numeric_dtype(series):

            if uniq > 50:
                score -= 3  # Penalizes continuous numeric targets

        if uniq == n_rows:
            score -= 10 ## likely ID column

        col_lower = str(col).lower()
        if "id" in col_lower:
            score -= 6  #penalize ID-like names

        if col == df.columns[-1]:
            score += 2 # small preference for last column

        if best_col is None or score > bestScore or (
            score == bestScore and uniq > df[best_col].nunique(dropna=True)
        ):
            bestScore = score
            best_col = col

    return best_col
#####################################################################
def is_classification_target(series: pd.Series) -> bool: #for determining  if a column is likely a classification target

    uniq = series.nunique()

    if uniq < 2:
        return False

    if series.dtype == "object" or str(series.dtype).startswith("category"):
        return True

    if pd.api.types.is_numeric_dtype(series):
        if uniq <= 25:
            return True # few unique numeric values = classification

    return False # Ootherwise regression
#####################################################################
def dataset_fingerprint(df: pd.DataFrame, target: str) -> str: #creates unique ID for the dataset
    
    cols = ",".join(df.columns.astype(str).tolist())
    shape = f"{df.shape[0]}x{df.shape[1]}"
    base = f"{shape}|{target}|{cols}"
    h = hashlib.md5(base.encode()).hexdigest()[:12]

    return f"fp_{h}"
#####################################################################
def profile_dataset(df: pd.DataFrame, target: str) -> Dict[str, Any]: #core EDA function

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset columns.")

    y = df[target]
    profile: Dict[str, Any] = {}

    profile["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    profile["columns"] = df.columns.astype(str).tolist()

    missing = (df.isna().mean() * 100).round(2).to_dict()
    profile["missing_pct"] = {str(k): float(v) for k, v in missing.items()}  # Ensure JSON safe types

    max_missing = max(profile["missing_pct"].values()) if profile["missing_pct"] else 0.0
    profile["max_missing_pct"] = max_missing

    profile["target"] = str(target)
    profile["target_dtype"] = str(y.dtype)
    profile["is_classification"] = bool(is_classification_target(y))
    print(f"[DEBUG-PROFILE] target={target} | is_classification={profile['is_classification']}")

    X = df.drop(columns=[target])

    numeric_cols = X.select_dtypes(include=["number"]).columns.astype(str).tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.astype(str).tolist()

    profile["feature_types"] = {
        "numeric": numeric_cols,
        "categorical": cat_cols,
    }

    profile["n_unique_by_col"] = {
        str(c): int(df[c].nunique(dropna=True))
        for c in df.columns.astype(str)
    }


    notes = []

    high_card_cols = [
        c for c, v in profile["n_unique_by_col"].items()
        if v > 50 and c != target
    ]

    profile["high_cardinality_features"] = high_card_cols

    if high_card_cols:
        notes.append("High-cardinality categorical features detected.")

    if profile["shape"]["rows"] < 1000:
        notes.append("Small dataset (<1000 rows): prefer simpler models / guard against overfitting.")

    if profile["shape"]["cols"] > 100:
        notes.append("High dimensionality (>100 columns): watch one-hot expansion and overfitting.")

    if len(cat_cols) > 0 and len(numeric_cols) == 0:
        notes.append("Categorical-only dataset detected.")

    if len(numeric_cols) > 0 and len(cat_cols) == 0:
        notes.append("Numeric-only dataset detected.")

    if max_missing >= 40:
        notes.append("Severe missing data detected (>40%).")

    elif max_missing >= 15:
        notes.append("Moderate missing data detected (>15%).")

    profile["notes"] = notes

    if profile["is_classification"]:
        vc = y.value_counts(dropna=False)

        profile["class_counts"] = {
            str(k): int(v) for k, v in vc.items()
        }

        profile["n_classes"] = int(len(vc))

        if len(vc) >= 2:
            ratio = float(vc.max() / max(vc.min(), 1))
        else:
            ratio = 1.0

        profile["imbalance_ratio"] = round(ratio, 3)

        if ratio >= 3.0:
            profile["notes"].append(
                "Imbalance detected (ratio >= 3.0): prioritise macro metrics / balanced accuracy."
            )

    else:
        profile["class_counts"] = None
        profile["imbalance_ratio"] = None
        profile["notes"].append(
            "Non-classification target detected: this template focuses on classification."
        )

    profile["dataset_complexity"] = {
        "n_features": len(numeric_cols) + len(cat_cols),
        "n_numeric": len(numeric_cols),
        "n_categorical": len(cat_cols),
    }

    return profile
