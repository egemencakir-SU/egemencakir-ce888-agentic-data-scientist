import os
import subprocess
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Add project root

from tools.data_profiler import profile_dataset, infer_target_column
from tools.modelling import build_preprocessor, select_models, train_models
from agents.planner import create_plan
from agents.reflector import reflect, should_replan
#####################################################################
def test_smoke_run():
    cmd = [
        sys.executable,
        "run_agent.py",
        "--data", "data/example_dataset.csv",
        "--target", "auto",
        "--quiet",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")  # Run full pipeline

    assert result.returncode == 0, f"Process failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert len(stdout_lines) > 0

    out_dir = stdout_lines[-1]
    assert os.path.isdir(out_dir)

    expected_files = [
        "report.md",
        "metrics.json",
        "reflection.json",
        "eda_summary.json",
        "plan.json",
        "confusion_matrix.png",
    ]

    for fname in expected_files:
        assert os.path.exists(os.path.join(out_dir, fname))
#####################################################################
def test_data_profiler_basic():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": ["x", "y", "x", "z"],
        "target": [0, 1, 0, 1],
    })

    target = infer_target_column(df)  # Should detect 'target'
    profile = profile_dataset(df, target)

    assert profile["shape"]["rows"] == 4
    assert profile["is_classification"] is True
    assert "numeric" in profile["feature_types"]
#####################################################################
def test_modelling_pipeline():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": ["x", "y", "x", "z"],
        "target": [0, 1, 0, 1],
    })

    profile = profile_dataset(df, "target")

    preprocessor = build_preprocessor(profile)  # Should build without error
    models = select_models(profile)

    assert preprocessor is not None
    assert len(models) > 0
#####################################################################
def test_planner_basic():
    profile = {
        "shape": {"rows": 100, "cols": 3},
        "feature_types": {"numeric": ["a"], "categorical": ["b"]},
        "n_unique_by_col": {"a": 100, "b": 3},
        "missing_pct": {"a": 0, "b": 0},
        "is_classification": True,
        "imbalance_ratio": 1.0,
        "class_counts": {"0": 50, "1": 50},
        "target": "target",
        "columns": ["a", "b", "target"],
    }

    plan = create_plan(profile)

    assert "train_models" in plan
    assert "evaluate" in plan
#####################################################################
def test_reflector_basic():
    dataset_profile = {
        "shape": {"rows": 100, "cols": 3},
        "imbalance_ratio": 1.0,
        "is_classification": True,
        "class_counts": {"0": 50, "1": 50},
        "missing_pct": {},
        "feature_types": {"categorical": [], "numeric": ["a"]},
        "n_unique_by_col": {},
    }

    evaluation = {
        "model": "TestModel",
        "balanced_accuracy": 0.6,
        "f1_macro": 0.6,
        "precision_macro": 0.6,
        "recall_macro": 0.6,
    }

    all_metrics = [
        evaluation,
        {
            "model": "DummyMostFrequent",
            "balanced_accuracy": 0.5,
            "f1_macro": 0.5,
            "precision_macro": 0.5,
            "recall_macro": 0.5,
        },
    ]  # Add dummy baseline for realistic reflection behavior

    reflection = reflect(dataset_profile, evaluation, all_metrics)

    assert "status" in reflection

    assert isinstance(should_replan(reflection), bool)
#####################################################################
def test_train_models_basic():

    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6],
        "b": ["x", "y", "x", "y", "x", "y"],
        "target": [0, 1, 0, 1, 0, 1],
    })

    profile = profile_dataset(df, "target")
    preprocessor = build_preprocessor(profile)
    models = select_models(profile)

    results = train_models(
        df,
        target="target",
        preprocessor=preprocessor,
        candidates=models,
        seed=42,
        test_size=0.3,
        output_dir=".",
        verbose=False,
    )

    assert "best" in results
    assert len(results["all_metrics"]) > 0
#####################################################################
def test_evaluation_basic():
    from tools.evaluation import evaluate_best

    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6],
        "b": ["x", "y", "x", "y", "x", "y"],
        "target": [0, 1, 0, 1, 0, 1],
    })

    profile = profile_dataset(df, "target")
    preprocessor = build_preprocessor(profile)
    models = select_models(profile)

    results = train_models(
        df,
        target="target",
        preprocessor=preprocessor,
        candidates=models,
        seed=42,
        test_size=0.3,
        output_dir=".",
        verbose=False,
    )

    eval_payload = evaluate_best(results, out_dir=".")

    assert "best_metrics" in eval_payload
    assert "classification_report" in eval_payload
#####################################################################
def test_replan_strategy():
    from agents.reflector import apply_replan_strategy

    plan = ["train_models", "evaluate", "reflect"]

    reflection = {
        "replan_recommended": True,
        "suggestions": ["use_stratified_split"],
    }

    dataset_profile = {
        "shape": {"rows": 100, "cols": 3},
        "feature_types": {"numeric": ["a"], "categorical": ["b"]},
    }

    new_plan, updated_profile = apply_replan_strategy(plan, dataset_profile, reflection)  # Function returns tuple

    assert isinstance(new_plan, list)
    assert len(new_plan) >= len(plan)

    assert isinstance(updated_profile, dict) 

#####################################################################
def test_dataset_fingerprint():
    from tools.data_profiler import dataset_fingerprint

    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
    })

    fp = dataset_fingerprint(df, target="a") 

    assert isinstance(fp, str)
    assert len(fp) > 0  # Ensure fingerprint string exists
#####################################################################
def test_write_markdown_report():
    from tools.evaluation import write_markdown_report

    class DummyCtx:
        run_id = "test"
        started_at = "now"
        data_path = "test.csv"
        target = "target"

    eval_payload = {
        "best_metrics": {
            "model": "TestModel",
            "accuracy": 0.8,
            "balanced_accuracy": 0.8,
            "f1_macro": 0.8,
            "precision_macro": 0.8,
            "recall_macro": 0.8,
        },
        "all_metrics": [],
        "confusion_matrix_path": "cm.png",
    }

    write_markdown_report(
        out_path="test_report.md",
        ctx=DummyCtx(),
        fingerprint="abc",
        dataset_profile={
            "shape": {"rows": 10, "cols": 3},
            "feature_types": {"numeric": ["a"], "categorical": []},
            "notes": [],
        },
        plan=["train", "evaluate"],
        eval_payload=eval_payload,
        reflection={"suggestions": []},
    )

    assert os.path.exists("test_report.md")
#####################################################################
def test_infer_target_edge_cases():
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4, 5, 6],
    })

    target = infer_target_column(df)

    assert target in df.columns  # Should fallback to a column
#####################################################################
def test_planner_missing_and_imbalance():
    profile = {
        "shape": {"rows": 200, "cols": 4},
        "feature_types": {"numeric": ["a"], "categorical": ["b"]},
        "n_unique_by_col": {"b": 100},
        "missing_pct": {"a": 60},  # Trigger missing logic
        "is_classification": True,
        "imbalance_ratio": 5.0,  # Trigger imbalance logic
        "class_counts": {"0": 180, "1": 20},
        "target": "target",
        "columns": ["a", "b", "target"],
    }

    plan = create_plan(profile)

    assert isinstance(plan, list)
    assert len(plan) > 0
#####################################################################
def test_reflector_low_performance():
    dataset_profile = {
        "shape": {"rows": 100, "cols": 3},
        "imbalance_ratio": 5.0,
        "is_classification": True,
        "class_counts": {"0": 90, "1": 10},
        "missing_pct": {},
        "feature_types": {"categorical": [], "numeric": ["a"]},
        "n_unique_by_col": {},
    }

    evaluation = {
        "model": "BadModel",
        "balanced_accuracy": 0.4,
        "f1_macro": 0.3,
        "precision_macro": 0.3,
        "recall_macro": 0.3,
    }

    reflection = reflect(dataset_profile, evaluation, [evaluation])

    assert reflection["status"] in ["poor", "average", "good", "needs_attention"]
#####################################################################
def test_should_replan_trigger():
    reflection = {
        "status": "poor",
        "suggestions": ["fix imbalance"],
        "replan_recommended": True,
    }

    result = should_replan(reflection)

    assert isinstance(result, bool)
#####################################################################
def test_memory_guidance_in_planner():
    profile = {
        "shape": {"rows": 100, "cols": 3},
        "feature_types": {"numeric": ["a"], "categorical": ["b"]},
        "n_unique_by_col": {},
        "missing_pct": {},
        "is_classification": True,
        "imbalance_ratio": 1.0,
        "class_counts": {"0": 50, "1": 50},
        "target": "target",
        "columns": ["a", "b", "target"],
    }

    memory_hint = {
        "best_model": "RandomForest",
        "notes": ["worked well before"]
    }

    plan = create_plan(profile, memory_hint=memory_hint)  # Trigger memory guidance

    assert isinstance(plan, list)
    assert len(plan) > 0
#####################################################################
def test_should_replan_branches():
    # Case 1: recommended
    r1 = {"replan_recommended": True}
    assert isinstance(should_replan(r1), bool)

    # Case 2: no recommendation
    r2 = {"replan_recommended": False}
    assert isinstance(should_replan(r2), bool)

    # Case 3: missing key
    r3 = {}
    assert isinstance(should_replan(r3), bool)
#####################################################################
def test_save_json():
    from tools.evaluation import save_json

    data = {"a": 1, "b": 2}
    path = "test_json_output.json"

    save_json(path, data)

    assert os.path.exists(path)
#####################################################################
def test_planner_edge_cases():
    profile = {
        "shape": {"rows": 5, "cols": 2},  # extremely small dataset
        "feature_types": {"numeric": ["a"], "categorical": []},
        "n_unique_by_col": {},
        "missing_pct": {"a": 0},
        "is_classification": True,
        "imbalance_ratio": 10.0,  # extreme imbalance
        "class_counts": {"0": 4, "1": 1},
        "target": "target",
        "columns": ["a", "target"],
    }

    plan = create_plan(profile)

    assert isinstance(plan, list)
    assert len(plan) > 0
#####################################################################
def test_planner_dataset_size_strategy():
    profile = {
        "shape": {"rows": 10000, "cols": 5},  # large dataset trigger
        "feature_types": {"numeric": ["a"], "categorical": ["b"]},
        "n_unique_by_col": {},
        "missing_pct": {},
        "is_classification": True,
        "imbalance_ratio": 1.0,
        "class_counts": {"0": 5000, "1": 5000},
        "target": "target",
        "columns": ["a", "b", "target"],
    }

    plan = create_plan(profile)

    assert isinstance(plan, list)
#####################################################################    
def test_infer_target_realistic():
    df = pd.DataFrame({
        "age": [20, 25, 30, 22],
        "salary": [2000, 3000, 4000, 2500],
        "churn": [0, 1, 0, 1],
    })

    target = infer_target_column(df)

    assert target == "churn"