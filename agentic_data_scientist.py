import os
import json
import uuid
import traceback
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from agents.planner import create_plan
from agents.reflector import reflect, should_replan, apply_replan_strategy
from agents.memory import JSONMemory
from tools.data_profiler import profile_dataset, infer_target_column, dataset_fingerprint
from tools.modelling import build_preprocessor, select_models, train_models
from tools.evaluation import evaluate_best, write_markdown_report, save_json
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
#####################################################################
@dataclass  #holds every important metadata for each agent to use
class RunContext:
    run_id: str  
    started_at: str
    data_path: str
    target: str
    output_dir: str
    seed: int
    test_size: float
    max_replans: int
#####################################################################
def now_iso() -> str:  #for creating standart timestamp. it will be used for naming the output folders 
    return datetime.now(UTC).replace(microsecond=0).isoformat()
#####################################################################
class AgenticDataScientist:
    def __init__(self, memory_path: str = "agent_memory.json", verbose: bool = True):  #this constructor prepares the agent when an object gets created
        self.verbose = verbose                 #  Verbose controls logging output
        self.memory = JSONMemory(memory_path)  # past model experience info
        self.ctx: Optional[RunContext] = None  # Run metadata is stored here
        self.state: Dict[str, Any] = {}        # Transient execution state lives here (how many retry, replan mistakes etc.)
#####################################################################
    def log(self, msg: str) -> None:  #automatic printing helper function. might get removed 
        if self.verbose:
            print(f"[AgenticDataScientist] {msg}")
#####################################################################
    def loadData(self, path: str) -> pd.DataFrame:  #reads the dataset from folder. supports both csv and excel format. also repairs the damaged data

        self.log(f"Loading dataset: {path}") #logs which data is loaded 

        lowerPath = path.lower()  #converts path name to lowercase for safety reasons

        if lowerPath.endswith(".csv"):  
            df = pd.read_csv(path)  #csv read format

        elif lowerPath.endswith(".xlsx") or lowerPath.endswith(".xls"):
            df = pd.read_excel(path) #xlsx read format

            if df.shape[1] == 1:  #if excel has single column, check for speerators
                first_col = str(df.columns[0])

                if ";" in first_col:
                    print("Splitting single Excel column by semicolon")

                    split_df = df.iloc[:, 0].astype(str).str.split(";", expand=True)
                    split_df.columns = [c.strip() for c in first_col.split(";")]
                    df = split_df

        else:
            print("error in loadData")
            raise ValueError("Unsupported file format. Please provide a .csv, .xlsx, or .xls file.")

        df.columns = [str(c).strip() for c in df.columns]

        self.log(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols") #final dataset shape
        return df
#####################################################################
    def buildOutputDirectory(self, output_root: str) -> Tuple[str, str]:  #creates unique folder for every run

        run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8] #timestamp + random suffix
        output_dir = os.path.join(output_root, run_id)                                     #builds output path
        os.makedirs(output_dir, exist_ok=True)                                             #creates directory if it does not exist

        #print(run_id)
        #print(output_dir)
        return run_id, output_dir
#####################################################################
    def initContext(self, data_path: str, target: str, output_root: str, seed: int, test_size: float, max_replans: int) -> None: #for initializing everything before pipeline starts

        run_id, output_dir = self.buildOutputDirectory(output_root) # Generates run ID and output folder

        self.ctx = RunContext(
            run_id=run_id,
            started_at=now_iso(),
            data_path=data_path,
            target=target,
            output_dir=output_dir,
            seed=seed,
            test_size=test_size,
            max_replans=max_replans,
        )
        self.state = {
            "replan_count": 0,
            "retry_count": 0,
            "errors": [],
            "plan_history": [],
        }
#####################################################################
    def errorRecord(self, stage: str, exc: Exception) -> None: #keeps track of errors
    
        errPayload = {
            "timestamp": now_iso(),
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        self.state.setdefault("errors", []).append(errPayload)

        if self.ctx is not None:
            save_json(os.path.join(self.ctx.output_dir, "errors.json"), self.state["errors"])  # Persist errors immediately to disk

#####################################################################
    def targetResolve(self, df: pd.DataFrame, target: str) -> str: #for handling target column selection logic

        if target.strip().lower() != "auto":  #auto mode
            return target

        inferred = infer_target_column(df)
        
        if not inferred:
            print("error in targetResolve")
            raise ValueError("Could not infer target column. Please provide --target <name>.")
        
        self.log(f"Inferred target: {inferred}")
        return inferred
#####################################################################
    def saveCoreArtefacts(self, profile: Dict[str, Any], plan: List[str], eval_payload: Dict[str, Any], reflection: Dict[str, Any]) -> None: #saves all important outputs into json files

        save_json(os.path.join(self.ctx.output_dir, "eda_summary.json"), profile)  # Save dataset profile
        save_json(os.path.join(self.ctx.output_dir, "plan.json"), {"plan": plan})  # Save execution plan
        save_json(os.path.join(self.ctx.output_dir, "metrics.json"), eval_payload)  # Save evaluation metrics
        save_json(os.path.join(self.ctx.output_dir, "reflection.json"), reflection)  # Save reflection output

#####################################################################
    def updateMemory(self, fingerprint: str, profile: Dict[str, Any], eval_payload: Optional[Dict[str, Any]] = None, reflection: Optional[Dict[str, Any]] = None) -> None:  #for learning from past runs

        record: Dict[str, Any] = {
            "last_seen": now_iso(),
            "target": self.ctx.target,
            "shape": profile.get("shape"),
            "is_classification": profile.get("is_classification"),
            "replan_count": self.state.get("replan_count", 0),
        }

        prevRec = self.memory.getDatasetRecord(fingerprint) or {}
        history = list(prevRec.get("history", []))

        if eval_payload is not None:
            best = eval_payload["best_metrics"] #gets best model

            record["best_model"] = best["model"]
            record["best_metrics"] = best

            history.append({  #performance history
                "timestamp": now_iso(),
                "best_model": best["model"],
                "f1_macro": best.get("f1_macro"),
            })

        record["history"] = history

        if reflection is not None:
            record["last_reflection_status"] = reflection.get("status") # Stores reflection result either good or poor

        self.memory.upsertDatasetRecord(fingerprint, record)  # Save/update memory entry
#####################################################################
    def planModifications(self, candidates, profile, plan): #planner actually changes the ML pipeline by modifying the models

        candidates = list(candidates) # Makes a copy for security so we don’t mutate original list

        if "prefer_simpler_models" in plan: # if simple model, use LR, NB and Dummy
            simple = [c for c in candidates if c[0] in ["LogisticRegression", "NaiveBayes", "DummyMostFrequent"]]
            if len(simple) >= 2:
                candidates = simple

        if "try_more_models" in plan:
            rows = int(profile.get("shape", {}).get("rows", 0))

            if rows < 10000: # Adds heavier models for smaller datasets
                extra = [
                    ("ExtraTrees_Extra", ExtraTreesClassifier(n_estimators=400, random_state=self.ctx.seed)),
                    ("RF_Extra", RandomForestClassifier(n_estimators=500, random_state=self.ctx.seed))
                ]
            else:
                extra = [  # Limits complexity for large datasets or it takes ages to complete
                    ("RF_Extra", RandomForestClassifier(n_estimators=300, random_state=self.ctx.seed))
                ]

            existingNames = {name for name, _ in candidates}

            for name, model in extra:
                if name not in existingNames:
                    candidates.append((name, model))

            existingNames = {name for name, _ in candidates}

        if any(step in plan for step in [
            "handle_imbalance",
            "use_strong_imbalance_strategy",
            "consider_imbalance_strategy",
            "prioritise_class_weighted_models"
        ]):
            for i, (name, model) in enumerate(candidates):
                if hasattr(model, "get_params") and "class_weight" in model.get_params():
                    try:
                        model.set_params(class_weight="balanced") #auto class wrighting
                    except Exception:
                        pass
                    candidates[i] = (name, model)

   
        if "warn_extremely_small_dataset" in plan:
            candidates = [c for c in candidates if c[0] not in ["GradientBoosting", "SVC_RBF"]]

        if "retry_with_simpler_models" in plan:
            simple = [c for c in candidates if c[0] in ["LogisticRegression", "NaiveBayes", "DummyMostFrequent"]]
            if len(simple) >= 1:
                candidates = simple

        return candidates
#####################################################################
    def nonClassificationReport(self, fingerprint: str, profile: Dict[str, Any], plan: List[str]) -> None:  #if target is not classification, pipeline stops and report gets created
 
        notes = profile.get("notes", [])  # Read notes safely from the profile
        notes_text = "\n".join(f"- {note}" for note in notes) if notes else "- (none)"
        plan_text = "\n".join(f"- {step}" for step in plan)

        md = f"""# Agentic Data Scientist Report

**Run ID:** `{self.ctx.run_id}`  
**Started (UTC):** {self.ctx.started_at}  
**Dataset:** `{self.ctx.data_path}`  
**Target:** `{self.ctx.target}`  
**Fingerprint:** `{fingerprint}`  

## Dataset Profile
- Rows: **{profile.get("shape", {}).get("rows")}**
- Columns: **{profile.get("shape", {}).get("cols")}**
- Classification: **{profile.get("is_classification")}**
- Target dtype: **{profile.get("target_dtype")}**

## Plan
{plan_text}

## Outcome
This template run stopped before model training because the detected target is not classification-oriented.
The current coursework template and modelling pipeline are focused on classification tasks.

## Notes
{notes_text}
"""

        with open(os.path.join(self.ctx.output_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(md)
#####################################################################
    def applyPlanToCandidates(self, candidates, plan):
        priority = [s.split("prioritise_model:")[1] for s in plan if s.startswith("prioritise_model:")] #extract model names marked for priority

        if not priority:
            return candidates #if no priority, return as it is

        return sorted(candidates, key=lambda x: (x[0] not in priority)) #moves priority models to top of the list
#####################################################################
    def runTrainingCycle(self, df: pd.DataFrame, profile: Dict[str, Any], plan: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]: #ML Pileline execution
      
        if "build_preprocessor" not in plan:
            print("error in runTrainingCycle")
            self.log("Warning: plan missing step 'build_preprocessor'")

        if "select_models" not in plan:
            print("error in runTrainingCycle")
            raise ValueError("Plan is missing 'select_models'.")

        if "train_models" not in plan:
            print("error in runTrainingCycle")
            raise ValueError("Plan is missing 'train_models'.")

        if "evaluate" not in plan:
            print("error in runTrainingCycle")
            raise ValueError("Plan is missing 'evaluate'.")

        preprocessor = build_preprocessor(profile)  # Builds preprocessing pipeline (scaling, encoding, feature selection)

        candidates = select_models(profile, seed=self.ctx.seed) # Selects the  base ML models based on dataset profile
        candidates = self.planModifications(candidates, profile, plan)
        candidates = self.applyPlanToCandidates(candidates, plan)

        self.log(f"Candidate models: {[name for name, _ in candidates]}")
        self.log(f"Total models: {len(candidates)}")

        results = train_models( # Trains  all of the candidate models and collect the results
            df=df,
            target=self.ctx.target,
            preprocessor=preprocessor,
            candidates=candidates,
            seed=self.ctx.seed,
            test_size=self.ctx.test_size,
            output_dir=self.ctx.output_dir,
            verbose=self.verbose,
        )

        eval_payload = evaluate_best(results, out_dir=self.ctx.output_dir)  # Selects best model and computes evaluation metrics

        self.log(f"Best model: {eval_payload['best_metrics']['model']}")
        self.log(f"F1 macro: {eval_payload['best_metrics'].get('f1_macro')}")

        return results, eval_payload
#####################################################################
    def runReflection(self, profile: Dict[str, Any], eval_payload: Dict[str, Any], plan: List[str]) -> Dict[str, Any]: #Takes model results, looks at dataset characteristics and decides whether the result is good or bad
      
        if "reflect" not in plan:
            return {
                "status": "ok",
                "best_model": eval_payload["best_metrics"]["model"],
                "issues": [],
                "suggestions": [],
                "replan_recommended": False,
            }

        return reflect(
            dataset_profile=profile,
            evaluation=eval_payload["best_metrics"],
            all_metrics=eval_payload["all_metrics"],
        )
#####################################################################
    def exeOnce(self, df: pd.DataFrame, profile: Dict[str, Any], plan: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
      
        maxLives = 2  # One normal attempt plus one  retry

        for attempt in range(1, maxLives + 1):

            try:
                self.log(f"Execution attempt {attempt}/{maxLives}")  # Make retries visible in logs
                self.log("Starting training cycle")

                _, eval_payload = self.runTrainingCycle(df, profile, plan)
                self.log("Training complete")

                reflection = self.runReflection(profile, eval_payload, plan)
                self.log("Reflection complete")

                self.log(f"Reflection status: {reflection.get('status')}")
                self.log(f"Suggestions: {len(reflection.get('suggestions', []))}")
                return eval_payload, reflection

            except Exception as exc:
                self.state["retry_count"] += 1
                self.errorRecord(stage=f"execution_attempt_{attempt}", exc=exc)
                self.log(f"Execution attempt {attempt} failed: {exc}")

                if attempt >= maxLives:
                    raise

                if attempt == 2:
                    plan = plan + ["retry_with_simpler_models"]

                notes = list(profile.get("notes", []))
                notes.append("Retry used after execution failure. Keeping the same pipeline with safer orchestration path.")
                profile["notes"] = notes

        raise RuntimeError("Unexpected execution state reached.")
#####################################################################
#LOAD -> ANALYZE ->  PLAN -> EXECUTE ->  REFLECT ->  IMPROVE ->  SAVE
    def run(  
        self,
        data_path: str,
        target: str,
        output_root: str = "outputs",
        seed: int = 42,
        test_size: float = 0.2,
        max_replans: int = 1,
    ) -> str:

        self.initContext(
            data_path=data_path,
            target=target,
            output_root=output_root,
            seed=seed,
            test_size=test_size,
            max_replans=max_replans,
        )

        try:
            df = self.loadData(data_path)                                                            # Load dataset first so the rest of the run is data-driven
            self.ctx.target = self.targetResolve(df, target)                                         # Store the final target in context

            profile = profile_dataset(df, self.ctx.target)                                           # CreateS the dataset profile used by the planner
            self.log(f"Profile: rows={profile['shape']['rows']}, cols={profile['shape']['cols']}")

            fingerprint = dataset_fingerprint(df, self.ctx.target)                                   # Fingerprint links this dataset to memory

            prev = self.memory.getDatasetRecord(fingerprint)                                        # Memory hint is optional and soft

            if not prev and hasattr(self.memory, "findSimilarDataset"): # Try similar dataset if exact match not found
                prev = self.memory.findSimilarDataset(
                    target=self.ctx.target,
                    is_classification=profile.get("is_classification")
                )

            if prev:
                self.log(f"Memory hit: previously best={prev.get('best_model')} for fp={fingerprint}")
            else:
                self.log("No useful memory hint found for this dataset.")

            plan = create_plan(profile, memory_hint=prev)                 #   Planning should happen before execution

            if prev:
                if hasattr(self.memory, "getBestModel"):
                    best_model = self.memory.getBestModel(prev)
                else:
                    best_model = prev.get("best_model")

                if best_model:
                    plan.append(f"prioritise_model:{best_model}")

            self.state["plan_history"].append(plan)
            self.log(f"Initial plan: {plan}")

            if not profile.get("is_classification", False):

                self.log("Detected non-classification target (i.e. regression). Skipping classification training and finishing.")
                reflection = {
                    "status": "ok",
                    "best_model": None,
                    "issues": ["This template currently focuses on classification tasks."],
                    "suggestions": ["Extend tools/modelling.py and tools/evaluation.py for regression support if needed."],
                    "replan_recommended": False,
                }
                eval_payload = {
                    "best_metrics": {
                        "model": "not_applicable",
                        "accuracy": 0.0,
                        "balanced_accuracy": 0.0,
                        "f1_macro": 0.0,
                        "precision_macro": 0.0,
                        "recall_macro": 0.0,
                    },
                    "all_metrics": [],
                    "confusion_matrix_path": None,
                    "classification_report": "Not generated because the detected target is non-classification (i.e. regression).",
                }

                self.saveCoreArtefacts(profile, plan, eval_payload, reflection)
                self.nonClassificationReport(fingerprint, profile, plan)
                self.updateMemory(fingerprint, profile, eval_payload=None, reflection=reflection)
                self.log(f"Done. Outputs saved to: {self.ctx.output_dir}")
                
                return self.ctx.output_dir
#####################################################################
            for _ in range(self.ctx.max_replans + 1):
                
                eval_payload, reflection = self.exeOnce(df=df, profile=profile, plan=plan)  # Execute one cycle

                self.saveCoreArtefacts(profile, plan, eval_payload, reflection)  # Persist outputs after each cycle

                write_markdown_report(
                    out_path=os.path.join(self.ctx.output_dir, "report.md"),
                    ctx=self.ctx,
                    fingerprint=fingerprint,
                    dataset_profile=profile,
                    plan=plan,
                    eval_payload=eval_payload,
                    reflection=reflection,
                )

                self.updateMemory(fingerprint, profile, eval_payload=eval_payload, reflection=reflection)  # Learn from this run

                if not should_replan(reflection):
                    break  # Finishes when the reflector is satisfied

                if self.state["replan_count"] >= self.ctx.max_replans:
                    self.log("Replan suggested, but max_replans reached. Stopping.")
                    break

                self.state["replan_count"] += 1
                self.log(f"Replanning attempt #{self.state['replan_count']}...")

                plan, profile = apply_replan_strategy(plan, profile, reflection)  # Replans changes future execution state
                self.state["plan_history"].append(plan)
                self.log(f"Updated plan: {plan}")

            self.log(f"Done. Outputs saved to: {self.ctx.output_dir}")
            return self.ctx.output_dir

        except Exception as exc:
            self.errorRecord(stage="run", exc=exc)
            self.log(f"Run failed: {exc}")
            raise