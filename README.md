# 🛒 Customer Churn Prediction & Segmentation for Retention Strategy

**Author:** Nguyen Viet Trung Kien

**Date:** March 2026

**Tools:** Python, Pandas, NumPy, Scikit-Learn, Random Forest, XGBoost, PCA, K-Means, Matplotlib, Seaborn

---

# 📌 Project Overview

Customer churn is one of the most critical challenges in e-commerce because losing existing customers directly impacts revenue, profitability, and Customer Lifetime Value (CLV).

The objective of this project is to identify the main drivers of churn, build a predictive model to identify at-risk customers, and explore customer segmentation techniques to support targeted retention campaigns.

---

# 🎯 Business Problem

An e-commerce company observed that a significant number of customers stop purchasing after a short period of time.

The business wants to answer the following questions:

### 1. Customer Behavior Analysis

* What characteristics are commonly found among churned customers?
* Which factors contribute most to customer churn?

### 2. Churn Prediction

* Can machine learning accurately predict customers who are likely to churn?

### 3. Customer Segmentation

* Can churned customers be grouped into meaningful segments for personalized retention campaigns?

---

# 📂 Dataset

The dataset contains customer demographic information, purchasing behavior, engagement metrics, and service-related attributes.

| Metric          | Value |
| --------------- | ----- |
| Records         | 5,630 |
| Features        | 20    |
| Target Variable | Churn |
| Format          | XLSX  |

### Target Variable

| Value | Meaning           |
| ----- | ----------------- |
| 1     | Customer Churned  |
| 0     | Customer Retained |

---

# 🧱 Project Workflow

```text
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Hyperparameter Tuning
      ↓
Model Evaluation
      ↓
Feature Importance
      ↓
Customer Segmentation
      ↓
Business Recommendation
```

---

# 🔧 Data Preprocessing

## Missing Value Treatment

Several columns contained missing values.

```python
cols_missing = [
    'Tenure',
    'WarehouseToHome',
    'HourSpendOnApp',
    'OrderAmountHikeFromlastYear',
    'CouponUsed',
    'OrderCount',
    'DaySinceLastOrder'
]

for col in cols_missing:
    df[col].fillna(df[col].median(), inplace=True)
```

### Why Median?

Median is more robust to outliers and helps preserve the original distribution of customer behavior.

---

## Category Standardization

Different labels representing the same meaning were standardized.

```python
df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({
    'COD':'Cash on Delivery',
    'CC':'Credit Card'
})
```

---

## Feature Encoding

### One-Hot Encoding

```python
df_encoded = pd.get_dummies(
    df,
    columns=[
        'PreferredLoginDevice',
        'PreferredPaymentMode',
        'PreferedOrderCat',
        'MaritalStatus'
    ]
)
```

### Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_encoded['Gender'] = le.fit_transform(
    df_encoded['Gender']
)
```

---

## Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

# 📊 Exploratory Data Analysis

To identify churn drivers, Random Forest Feature Importance was used.

## Top 5 Churn Drivers

| Rank | Feature           |
| ---- | ----------------- |
| 1    | Tenure            |
| 2    | CashbackAmount    |
| 3    | WarehouseToHome   |
| 4    | Complain          |
| 5    | DaySinceLastOrder |

### Feature Importance Code

```python
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train_scaled,y_train)

importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
})

importance_df.sort_values(
    'Importance',
    ascending=False
)
```

---

# 💡 Key Business Insights

### Customers With Short Tenure Are More Likely To Churn

Most churned customers left within the first few months.

**Recommendation**

* Welcome rewards
* First purchase discounts
* Customer onboarding campaigns

---

### Lower Cashback Is Associated With Higher Churn

Customers receiving lower cashback rewards were more likely to leave.

**Recommendation**

* Tiered cashback programs
* Loyalty rewards

---

### Complaint History Strongly Predicts Churn

Customers who submitted complaints showed significantly higher churn probability.

**Recommendation**

* Faster complaint resolution
* Customer recovery programs

---

### Inactive Customers Are At Higher Risk

Customers with longer periods since their last purchase had a much higher churn rate.

**Recommendation**

* Re-engagement campaigns
* Personalized promotions

---

# 🤖 Churn Prediction Modeling

## Model Comparison

Several algorithms were evaluated.

| Model               | Recall |
| ------------------- | ------ |
| Logistic Regression | 0.329  |
| KNN                 | 0.470  |
| Gradient Boosting   | 0.517  |
| Random Forest       | 0.698  |

### Why Recall?

Missing a churned customer (False Negative) is more costly than targeting a retained customer.

Therefore Recall was chosen as the primary evaluation metric.

---

## Final Model

### Random Forest Classifier

```python
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train_scaled,y_train)

y_pred = rf.predict(X_test_scaled)
```

### Performance

| Metric                       | Score  |
| ---------------------------- | ------ |
| Validation Balanced Accuracy | 90.09% |
| Test Balanced Accuracy       | 88.55% |

---

## Hyperparameter Tuning

```python
param_grid = {
    'n_estimators':[50,100,200],
    'max_depth':[10,20,30],
    'min_samples_split':[2,5,10],
    'bootstrap':[True,False]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    scoring='balanced_accuracy',
    cv=5
)

grid_search.fit(
    X_train_scaled,
    y_train
)
```

---

# 👥 Customer Segmentation

Only churned customers were included in the clustering analysis.

## PCA

```python
pca = PCA(
    n_components=0.90
)

pca_data = pca.fit_transform(
    scaled_data
)
```

---

## K-Means Clustering

```python
kmeans = KMeans(
    n_clusters=4,
    random_state=42
)

clusters = kmeans.fit_predict(
    pca_data
)
```

---

## Results

### K-Means

* No clear elbow point was observed.

### Hierarchical Clustering

* Silhouette Score remained low.

### Conclusion

Customer groups were highly overlapping and did not form meaningful clusters.

Additional behavioral features such as browsing activity, campaign engagement, and session frequency may improve segmentation performance.

---

# 💡 Business Recommendations

## Short-Term Actions

### Improve Onboarding Experience

* Welcome rewards
* Personalized onboarding
* First-purchase incentives

### Strengthen Loyalty Programs

* Cashback optimization
* Tiered reward systems

### Improve Complaint Resolution

* Faster support response
* Customer recovery workflows

### Re-Engage Inactive Customers

* Email campaigns
* Push notifications
* Personalized discounts

---

## Long-Term Actions

* Build a Churn Early Warning System (EWS)
* Collect additional behavioral data
* Monitor model performance regularly
* Retrain models periodically

---

# 📌 Key Results

✅ Identified Top 5 Customer Churn Drivers

✅ Built a Random Forest Churn Prediction Model

✅ Achieved 88.55% Balanced Accuracy

✅ Generated Actionable Retention Recommendations

✅ Evaluated Customer Segmentation Feasibility

✅ Completed an End-to-End Machine Learning Workflow

---
