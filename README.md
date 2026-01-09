


#  Bank Term Deposit Subscription Prediction

##  Project Overview

Banks conduct marketing campaigns to encourage customers to subscribe to term deposits. However, contacting every customer is inefficient and costly. This project builds an **end-to-end machine learning solution** to predict whether a customer is likely to subscribe to a term deposit, enabling **targeted and data-driven marketing decisions**.

The project follows a complete data science workflow including data preprocessing, feature engineering, model building, evaluation, and business-oriented model selection.

---

##  Problem Statement

The key challenge is to identify customers who are most likely to subscribe to a term deposit so that banks can:

* Reduce marketing costs
* Increase conversion rates
* Improve customer targeting

This is formulated as a **binary classification problem**.

---

##  Dataset Overview

The dataset contains customer demographic, financial, and campaign-related information such as:

* Age, job, education, marital status
* Account balance and loan information
* Marketing contact details and duration
* Previous campaign outcomes

**Target Variable:**

* `y` → Whether the customer subscribed to a term deposit (`yes` / `no`)

---

##  Project Objective

* Build a predictive model to identify potential subscribers
* Handle class imbalance effectively
* Compare multiple machine learning models
* Select a final model based on **business impact**, not just accuracy

---

##  Success Criteria

* High **recall** for subscribers (positive class)
* Balanced **F1-score**
* Interpretability for business understanding
* Scalable and reusable ML pipeline

---

##  Project Workflow

### 1️ Data Understanding & Preprocessing

* Dataset inspection and cleaning
* Duplicate and missing value checks
* Outlier handling using IQR-based capping

### 2️ Feature Engineering

* One-Hot Encoding for categorical variables
* Log transformation for skewed numerical features
* Feature selection using correlation analysis

### 3️ Data Scaling & Splitting

* StandardScaler for normalization
* 80:20 train-test split with stratification

### 4️ Handling Imbalanced Data

* Applied **SMOTE** to balance subscriber and non-subscriber classes

---

## Machine Learning Models Implemented

| Model               | Description                        |
| ------------------- | ---------------------------------- |
| Logistic Regression | Baseline and tuned model           |
| Decision Tree       | Captures non-linear relationships  |
| Random Forest       | Ensemble model for higher accuracy |

---

##  Model Evaluation Metrics

* Accuracy
* Precision
* Recall (**primary business metric**)
* F1-score
* Confusion Matrix

---

##  Model Comparison Summary

| Model                       | Accuracy | Recall (Subscribers) | F1-score |
| --------------------------- | -------- | -------------------- | -------- |
| Logistic Regression (Tuned) | 82%      | **0.79**             | **0.50** |
| Decision Tree (Tuned)       | 84%      | 0.57                 | 0.45     |
| Random Forest (Tuned)       | 88%      | 0.47                 | 0.48     |

---

##  Final Model Selection

**Tuned Logistic Regression** was selected as the final model because:

* It achieved the **highest recall** for subscribers
* It minimized missed potential customers
* It aligned best with business objectives
* It is interpretable and stable

> In marketing use cases, missing a potential subscriber results in lost revenue, making recall more important than raw accuracy.

---

##  Model Explainability

* Logistic Regression coefficients were used to understand feature importance
* Key influential features included:

  * Call duration
  * Previous campaign outcome
  * Campaign contact frequency
  * Account balance and loan status

---

##  Future Work (Optional)

* Deploy the model using **Flask or FastAPI**
* Integrate real-time data using streaming systems
* Explore advanced models like **XGBoost / LightGBM**
* Apply cost-sensitive learning
* Retrain models periodically with new data

---

##  Distributed ML Perspective

Although this project uses a moderate-sized dataset, the same pipeline can be scaled using:

* **Apache Spark** for distributed data processing
* **Spark MLlib** for large-scale model training
* **Spark Streaming** for real-time transaction analysis

---

##  Key Learnings

* Business metrics matter more than accuracy
* Feature engineering directly impacts model performance
* Simpler models can outperform complex ones in real-world scenarios
* Interpretability is crucial in financial applications

---

##  Tech Stack

* Python
* Pandas, NumPy, Matplotlib
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Joblib

---

##  Repository Structure

```
├── data/
│   └── bank.csv
├── notebooks/
│   └── bank_term_deposit_prediction.ipynb
├── model/
│   └── final_logistic_regression_model.joblib
├── README.md
```

---

##  Final Note

This project demonstrates a **complete, business-focused machine learning pipeline** with strong emphasis on model evaluation, interpretability, and real-world applicability.

