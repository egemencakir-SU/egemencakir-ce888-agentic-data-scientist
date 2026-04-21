# Dataset Documentation

This project uses four different datasets to evaluate the behaviour of the Agentic Data Scientist under varying conditions. Each dataset represents a different type of problem or data characteristic.

---

## 1. example_dataset.csv

A small synthetic classification dataset used for testing basic functionality.

- **Type:** Classification  
- **Size:** Very small  
- **Features:** Mixed (simple numeric/categorical)  
- **Purpose:**  
  Used to evaluate system behaviour under extreme data scarcity. Helps verify whether the system can still build and evaluate models on minimal data.

---

## 2. dataset_facebook.xlsx

A real-world dataset containing information related to Facebook posts.

- **Type:** Classification  
- **Features:** Mixed (numeric + categorical)  
- **Characteristics:**
  - High-cardinality categorical features  
  - Multiple classes  
  - Moderate dataset size  
- **Purpose:**  
  Used to test how the system handles more complex data, including high dimensionality and categorical encoding challenges.

---

## 3. WineQuality.csv

A dataset containing physicochemical properties of wine samples.

- **Type:** Originally regression (treated as classification candidate)  
- **Features:** Numeric only  
- **Target:** Quality score (continuous/discrete numeric)  
- **Purpose:**  
  Used to test whether the system can correctly identify when a dataset is not suitable for classification and stop the pipeline accordingly.

---

## 4. bank.csv

A dataset related to bank marketing campaigns.

- **Type:** Binary classification  
- **Target:** `deposit` (yes/no)  
- **Features:** Mixed (numeric + categorical)  
- **Characteristics:**
  - Medium to large size  
  - Real-world structured data  
  - Relatively balanced classes  
- **Purpose:**  
  Used to evaluate system performance on a realistic classification task with mixed feature types and practical complexity.

---

## Summary

These datasets were selected to cover:

- Small vs large datasets  
- Simple vs complex feature structures  
- Classification vs non-classification scenarios  
- Low vs high cardinality  

This variety allows the system’s adaptability and decision-making behaviour to be properly evaluated.