from typing import Any, Dict, List, Optional
#####################################################################
def KEEP_ORDER(items: List[str]) -> List[str]:  #used for removing the duplicate steps 

    seen = set()  #keeps track of seen steps
    ordered: List[str] = []  #the order of steps

    for item in items:

        if item not in seen:  #adds unique steps into order
            ordered.append(item)
            seen.add(item)
    #print(####################)
    #print(ordered)
    return ordered
#####################################################################
def insertBFR(plan: List[str], anchor: str, step: str) -> None: #inserts a step before an anchor

    if step in plan:
        
        return

    if anchor in plan:
        plan.insert(plan.index(anchor), step)  #puts step before ancor

    else:
        plan.append(step)
#####################################################################
def maxMissingPercentage(missing_pct: Dict[str, Any]) -> float: #extracts the max missing percentage

    if not missing_pct:
        return 0.0


    vals: List[float] = []

    for value in missing_pct.values(): #json values might be string 
        try:
            vals.append(float(value)) #float conversion 

        except (TypeError, ValueError):
            print("error in maxMissingPercentage function")
            continue

    return max(vals) if vals else 0.0
#####################################################################
def C_High_Cardinality_Categoricals(categorical_cols: List[str], n_unique_by_col: Dict[str, Any], threshold: int = 50) -> int: #Counts categorical columns with high number of unique values

    count = 0

    for col in categorical_cols:
        try:
            if int(n_unique_by_col.get(col, 0)) > threshold:
                count += 1

        except (TypeError, ValueError):
            print("error in C_High_Cardinality_Categoricals function")
            continue

    #print(####################)
    #print(count)
    return count
############################################################################################3
def BASE_PLAN() -> List[str]: #this is the core pipeline skeleton
    return [                   #profile -> preprocess -> train -> evaluate -> reflect
        "profile_dataset",
        "validate_target_and_schema",
        "assess_data_quality",
        "select_evaluation_strategy",
        "build_preprocessor",
        "select_models",
        "train_models",
        "evaluate",
        "reflect",
        "write_report",
        "persist_memory",
    ]
##############################################################################################
def datasetSizeStrategy(plan: List[str], rows: int, cols: int) -> None:

    if rows < 500:
        insertBFR(plan, "train_models", "prefer_simpler_models")                       #small dataset might get overfitted

    elif rows >= 5000:
        insertBFR(plan, "select_models", "control_training_cost")                      #big dataset -> training can be coslty

    if cols > 100:
        insertBFR(plan, "build_preprocessor", "handle_high_dimensional_features")      #too much feature will result overfitting
###################################################################
def featureTypeStrategy(plan: List[str], numeric_cols: List[str], categorical_cols: List[str], n_unique_by_col: Dict[str, Any]) -> None:

    if numeric_cols and categorical_cols:                                              #if numeric and categoric, mixed type preprocessing needs to be done.
        insertBFR(plan, "build_preprocessor", "use_mixed_type_preprocessing")

    elif numeric_cols: 
        insertBFR(plan, "build_preprocessor", "use_numeric_focused_preprocessing")

    elif categorical_cols:
        insertBFR(plan, "build_preprocessor", "use_categorical_focused_preprocessing")

    highCardCount = C_High_Cardinality_Categoricals(categorical_cols, n_unique_by_col, threshold=50)
    
    if highCardCount > 0:                                                               #too much feature
        insertBFR(plan, "build_preprocessor", "detect_high_cardinality_categoricals")
        insertBFR(plan, "build_preprocessor", "reduce_one_hot_risk")
#####################################################################
def missingDataStrategy(plan: List[str], max_missing_pct: float) -> None:
    
    if max_missing_pct >= 40.0:                                              #severe loss of data
        insertBFR(plan, "build_preprocessor", "handle_severe_missing_data")
    
    elif max_missing_pct >= 15.0:                                             #moderate loss of data
        insertBFR(plan, "build_preprocessor", "handle_moderate_missing_data")
    
    elif max_missing_pct > 0.0:                                               #light loss of data
        insertBFR(plan, "build_preprocessor", "handle_light_missing_data")
##############################################################################
def classificationStrategy(plan: List[str], is_classification: bool, imbalance_ratio: Optional[float], class_counts: Optional[Dict[str, Any]]) -> None: #decides whether the problem is a classification. and if it is, what kind of classification?
    
    if not is_classification:                                                                    #if its not a classification do not traing
        insertBFR(plan, "select_evaluation_strategy", "detect_non_classification_target")

        return

    insertBFR(plan, "select_evaluation_strategy", "confirm_classification_task")
    insertBFR(plan, "select_evaluation_strategy", "prefer_macro_and_balanced_metrics")          #accuracy is not enough, more metrics are added

    nClasses = len(class_counts or {})
    
    if nClasses > 2:                                                                           #if there are multiple classes, pinary != multiclass. macro F1 is important
        insertBFR(plan, "select_evaluation_strategy", "prepare_multiclass_evaluation")
 
    imbalance = float(imbalance_ratio or 1.0)                                      #imbalance control
    
    if imbalance >= 5.0:                                                           #serious imbalance
        insertBFR(plan, "select_models", "use_strong_imbalance_strategy")
        insertBFR(plan, "train_models", "prioritise_class_weighted_models")
        insertBFR(plan, "evaluate", "inspect_minority_class_performance")
    
    elif imbalance >= 3.0:                                                         #moderate imbalance
        insertBFR(plan, "select_models", "consider_imbalance_strategy")
        insertBFR(plan, "train_models", "prioritise_balanced_models")
        insertBFR(plan, "evaluate", "inspect_minority_class_performance")
#####################################################################
def memoryGuidance(plan: List[str], memory_hint: Optional[Dict[str, Any]]) -> None: #learns from previous runs
    
    if not memory_hint:                                                        #if there's no previous memory do nothing
        return

    bestModel = memory_hint.get("best_model")
    bestMetrics = memory_hint.get("best_metrics", {})

    if bestModel:
        insertBFR(plan, "train_models", f"prioritise_model:{bestModel}")        #prioritizes best model

    try:
        priorBalancedACC = float(bestMetrics.get("balanced_accuracy", 0.0))        #performance check
    
    except (TypeError, ValueError):
        print("error in memoryGuidance")
        priorBalancedACC = 0.0

    if priorBalancedACC >= 0.80:                                              #if prior run had success accuracy of 80 or above, just use it.
        insertBFR(plan, "select_models", "reuse_successful_strategy_hint")
#####################################################################
def edgeCaseStrategy(plan: List[str], rows: int) -> None: #will be only used for very small datasets

    if rows <= 50:
        #print("small data")
        insertBFR(plan, "train_models", "warn_extremely_small_dataset")
#####################################################################
def create_plan(dataset_profile: Dict[str, Any], memory_hint: Optional[Dict[str, Any]] = None) -> List[str]:

    shape = dataset_profile.get("shape", {})                                       #gets dataset info
    rows = int(shape.get("rows", 0))
    cols = int(shape.get("cols", 0))

    featureTypes = dataset_profile.get("feature_types", {})                       #column types are gathered
    numericCols = featureTypes.get("numeric", []) or []
    categoricalCols = featureTypes.get("categorical", []) or []

    uniqueCols = dataset_profile.get("n_unique_by_col", {}) or {}             #number of unique columns
    missingPercentage = dataset_profile.get("missing_pct", {}) or {}                     #missing data percentage
    maxMissing = maxMissingPercentage(missingPercentage)

    isClassification = bool(dataset_profile.get("is_classification", False))      #classification info
    imbalance_R = dataset_profile.get("imbalance_ratio")
    class_C = dataset_profile.get("class_counts")
    target = str(dataset_profile.get("target", ""))
    columns = dataset_profile.get("columns", []) or []

    plan = BASE_PLAN()  # Start from a stable common backbone

    datasetSizeStrategy(plan, rows, cols)                                            # big or small data?
    featureTypeStrategy(plan, numericCols, categoricalCols, uniqueCols)              # numeric or categorical
    missingDataStrategy(plan, maxMissing)                                  # any missing values?
    classificationStrategy(plan, isClassification, imbalance_R, class_C)             # classification? imbalance?
    memoryGuidance(plan, memory_hint)                                                # any previous memory?
    edgeCaseStrategy(plan, rows)                                                     # extreme case?

    plan = KEEP_ORDER(plan)  # cleanup for duplicate steps

    #print(###########################)
    print(plan)
    return plan