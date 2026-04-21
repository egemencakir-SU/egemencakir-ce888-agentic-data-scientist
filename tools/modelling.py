from typing import Any, Dict, List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
#####################################################################
def safeFloat(value: Any, default: float = 0.0) -> float: #converts any value to float.

    try:
        return float(value)

    except (TypeError, ValueError):
        return default
#####################################################################
def _hasHighCardinality(profile: Dict[str, Any], threshold: int = 50) -> bool: # Detects if categorical features have too many unique values

    feature_types = profile.get("feature_types", {}) or {}
    categorical_cols = feature_types.get("categorical", []) or []
    n_unique_by_col = profile.get("n_unique_by_col", {}) or {}

    for col in categorical_cols:
        try:
            if int(n_unique_by_col.get(col, 0)) > threshold:
                return True

        except (TypeError, ValueError):
            continue

    return False # No high cardinality found
#####################################################################
def build_preprocessor(profile: Dict[str, Any]) -> ColumnTransformer: # Builds preprocessing pipeline for numeric and categorical data

    num_cols = profile["feature_types"]["numeric"]     # Numeric columns
    cat_cols = profile["feature_types"]["categorical"] # Categorical columns

    maxMiss = 0.0 # Find maximum missing percentage across all columns

    for value in (profile.get("missing_pct", {}) or {}).values():
        try:
            maxMiss = max(maxMiss, float(value))
        except (TypeError, ValueError):
            continue

    numericInputerStrategy = "median" if maxMiss < 40.0 else "constant" # Chooses imputation strategy based on missing severity

    k = min(50, len(num_cols)) if len(num_cols) > 0 else "all" # Selects number of features for SelectKBest

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=numericInputerStrategy, fill_value=0)), # Fills the missing numeric values
        ("scaler", StandardScaler(with_mean=True)),                                # Normalizes numeric features
        ("select", SelectKBest(score_func=f_classif, k=k)),                        # Selects top k features
    ])


    try:  # Handles sklearn version compatibility for OneHotEncoder
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # New sklearn versions
    
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)  # Backward compatibility

    categoricalInputerStrategy = "most_frequent" if maxMiss < 40.0 else "constant"

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=categoricalInputerStrategy, fill_value="missing")),  # Fill missing categories
        ("onehot", ohe),  # converts categorical values to one-hot encoding
    ])

    return ColumnTransformer(  # Combine numeric and categorical pipelines
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )
#####################################################################
def select_models(profile: Dict[str, Any], seed: int = 42) -> List[Tuple[str, Any]]: # for selecting the appropriate ML models based on dataset characteristics
    
    rows = int(profile["shape"]["rows"])
    cols = int(profile["shape"]["cols"])
    imb = safeFloat(profile.get("imbalance_ratio") or 1.0, 1.0) # Class imbalance ratio

    hasHighCardinality = _hasHighCardinality(profile, threshold=50)

    class_weight = "balanced" if imb >= 3.0 else None  # handles imbalance automatically

    rf_trees = 100 if rows < 5000 else 150 # adjusts tree count based on dataset size since it takes too long to train on large datasets
    extra_trees = 100 if rows < 5000 else 150

    candidates: List[Tuple[str, Any]] = [
        ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")), # Baseline model
        ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight=class_weight)),
        ("RandomForest", RandomForestClassifier(
            n_estimators=rf_trees,
            random_state=seed,
            n_jobs=-1,
            class_weight=class_weight,
        )),
    ]

    if rows <= 10000 and cols <= 500:
        candidates.append(("GradientBoosting", GradientBoostingClassifier(random_state=seed)))

    if rows >= 500:
        candidates.append(("ExtraTrees", ExtraTreesClassifier(
            n_estimators=extra_trees,
            random_state=seed,
            n_jobs=-1,
            class_weight=class_weight,
        )))

    if rows <= 20000 and cols <= 200 and not hasHighCardinality:
        candidates.append(("SVC_RBF", SVC(kernel="rbf", probability=True, class_weight=class_weight)))

    if cols < 200:
        candidates.append(("NaiveBayes", GaussianNB()))

    if hasHighCardinality:
        candidates = [c for c in candidates if c[0] not in ["LogisticRegression", "NaiveBayes", "GradientBoosting"]]

    return candidates
#####################################################################
def train_models(   # main training loop for all candidate models
    df: pd.DataFrame,
    target: str,
    preprocessor: ColumnTransformer,
    candidates: List[Tuple[str, Any]],
    seed: int,
    test_size: float,
    output_dir: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.") #validate target

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    if pd.api.types.is_float_dtype(y):  #converts float target to classification if small number of unique values

        if y.nunique() <= 25:
            print("[DEBUG] Converting float target to discrete classes")
            y = y.round().astype(int)

    mask = ~y.isna()  # Removes missing target rows
    X = X.loc[mask]
    y = y.loc[mask]

    if y.nunique(dropna=True) < 2:
        raise ValueError("Target must contain at least two classes.")

    stratify = y if (y.nunique(dropna=True) > 1 and y.value_counts().min() >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    results: List[Dict[str, Any]] = []         #stores successful models
    failed_models: List[Dict[str, str]] = []   #tores failed models

    for name, model in candidates: #trains each model
        if verbose:
            print(f"[Modelling] Training: {name} (train={len(X_train)}, test={len(X_test)})")

        try:
            pipe = Pipeline(steps=[
                ("preprocess", preprocessor),  #apply preprocessing and model
                ("model", model),
            ])

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            metrics = {
                "model": name,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            }

            results.append({
                "name": name,
                "pipeline": pipe,
                "metrics": metrics,
                "X_test": X_test,
                "y_test": y_test,
                "y_pred": y_pred,
            })

        except Exception as exc:
            failed_models.append({"model": name, "error": str(exc)})

            if verbose:
                print(f"[Modelling] Failed: {name} -> {exc}")

    if not results:                                             #if all models failed crash
        raise RuntimeError(f"All models failed: {failed_models}")

    results.sort(  #sorst models by performance
        key=lambda r: (r["metrics"]["balanced_accuracy"], r["metrics"]["f1_macro"]),
        reverse=True,
    )

    return {
        "results": results,
        "best": results[0],
        "all_metrics": [r["metrics"] for r in results],
        "failed_models": failed_models,
        "split_summary": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "used_stratify": bool(stratify is not None),
        },
    }