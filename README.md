\# CE888 Agentic Data Scientist



\*\*GitHub Repository:\*\*  

https://github.com/egemencakir-SU/egemencakir-ce888-agentic-data-scientist



This project implements an offline Agentic Data Scientist capable of performing end-to-end classification tasks without relying on large language models. The system is designed to automatically analyse a dataset, generate an execution plan, train multiple machine learning models, evaluate their performance, and iteratively improve its decisions using reflection and memory.



The core of the system is the executor defined in `agentic\_data\_scientist.py`, which orchestrates the entire workflow. It manages data loading, target detection, dataset profiling, planning, model training, evaluation, reflection, and result storage. The pipeline is fully automated and designed to handle different datasets without manual intervention.



The architecture consists of four main components: Planner, Reflector, Memory, and Executor. The Planner generates a dynamic execution plan based on dataset characteristics such as size, feature types, missing values, and class imbalance. The Reflector analyses model performance using multiple metrics and determines whether improvements are needed. The Memory component stores past runs and allows the agent to reuse successful strategies. The Executor coordinates all components and ensures robust execution with retry logic and error handling.



The modelling pipeline is implemented using scikit-learn and includes multiple algorithms such as Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, Support Vector Machines, Naive Bayes, and a Dummy baseline. Model selection is adaptive and depends on dataset characteristics. Preprocessing is handled automatically using imputation, scaling, feature selection, and one-hot encoding depending on the data.



Evaluation is performed using multiple metrics including accuracy, balanced accuracy, macro F1-score, precision, and recall. The system also generates a confusion matrix and a classification report. Instead of relying on a single metric, the agent prioritises balanced accuracy and macro F1-score to ensure robust performance, especially for imbalanced datasets.



After each run, the agent generates structured outputs including dataset profiling information, execution plan, evaluation metrics, reflection results, and a markdown report. These outputs are stored in a timestamped directory inside the `outputs/` folder.



To install the required dependencies, run:



```bash

pip install -r requirements.txt

```



To execute the agent, use:



```bash

python run\_agent.py --data data/example\_dataset.csv --target auto

```



The `--target` argument can be set to a specific column name or to `auto`, in which case the system will attempt to infer the target column automatically based on heuristic rules.



The project also includes a sanity check script to verify that the pipeline runs correctly and produces all required outputs:



```bash

python tests/sanity\_check.py

```



For full testing with coverage, run:



```bash

pytest --cov=agents --cov=tools --cov-report=html tests/



```

To view the coverage report:



```bash

start htmlcov/index.html

```



The system has been extended beyond the base template with improved planning logic, enhanced reflection analysis, memory-based learning, adaptive model selection, and robust execution handling. These improvements allow the agent to better handle different dataset scenarios such as small datasets, class imbalance, and high-dimensional data.



This project demonstrates a modular and rule-based approach to building an autonomous data science system that can adapt its behaviour based on data characteristics and past experience, without relying on external AI models.



