# ❓ Question 1: What factors contribute to customer churn?

## 🎯 Objective

The first objective is to understand the characteristics and behaviors of churned customers and identify the key factors that contribute to customer churn.

Understanding these factors helps the business design more effective retention strategies and intervene before customers leave.

---

## 🔍 Analysis Approach

To answer this question:

### Step 1: Perform Data Cleaning

* Handle missing values
* Check duplicate records
* Standardize categorical values

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

---

### Step 2: Feature Importance Analysis

Random Forest was used to identify the most influential variables affecting churn.

```python
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train_scaled,y_train)

importance_df = pd.DataFrame({
    'Feature':X_train.columns,
    'Importance':rf.feature_importances_
})
```

---

## 📊 Results

### Top 5 Churn Drivers

| Rank | Feature           |
| ---- | ----------------- |
| 1    | Tenure            |
| 2    | CashbackAmount    |
| 3    | WarehouseToHome   |
| 4    | Complain          |
| 5    | DaySinceLastOrder |

---

## 💡 Key Findings

### Tenure

Most churned customers have very short tenure.

Approximately 80% of churned customers leave within the first five months.

#### Insight

The customer onboarding period is the most critical stage in the customer lifecycle.

#### Recommendation

* Improve onboarding experience
* Offer welcome campaigns
* Provide first-purchase incentives

---

### CashbackAmount

Churned customers generally receive lower cashback rewards.

#### Insight

Customers may perceive lower value from the platform when reward benefits are limited.

#### Recommendation

* Introduce tiered cashback programs
* Strengthen loyalty rewards
* Provide personalized promotions

---

### WarehouseToHome

Customers located farther from warehouses tend to churn more frequently.

#### Insight

Longer delivery times may reduce customer satisfaction.

#### Recommendation

* Improve logistics operations
* Offer real-time delivery tracking
* Optimize shipping experience

---

### Complain

Complaint history is strongly associated with churn.

#### Insight

Negative customer experiences increase churn probability.

#### Recommendation

* Reduce complaint resolution time
* Build customer recovery programs
* Monitor complaint rates continuously

---

### DaySinceLastOrder

Churned customers tend to have longer periods of inactivity.

#### Insight

Purchase inactivity is an early warning signal of churn.

#### Recommendation

* Launch re-engagement campaigns
* Send reminder emails
* Offer personalized discounts

---

# ❓ Question 2: Can Machine Learning Predict Customer Churn?

## 🎯 Objective

The second objective is to build a machine learning model that can accurately identify customers likely to churn.

This allows the company to proactively launch retention campaigns before customers leave.

---

## 🔍 Analysis Approach

### Step 1: Feature Engineering

Categorical variables were encoded using:

* One-Hot Encoding
* Label Encoding

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

---

### Step 2: Split Dataset

```python
X = df_encoded.drop('Churn',axis=1)
y = df_encoded['Churn']

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
```

---

### Step 3: Standardize Features

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### Step 4: Compare Models

Models tested:

* Logistic Regression
* KNN
* Gradient Boosting
* Random Forest

Evaluation Metric:

### Recall

Recall was prioritized because missing a churned customer (False Negative) is more costly than incorrectly targeting a retained customer.

---

## 📊 Results

### Model Comparison

| Model               | Recall |
| ------------------- | ------ |
| Logistic Regression | 0.329  |
| KNN                 | 0.470  |
| Gradient Boosting   | 0.517  |
| Random Forest       | 0.698  |

---

### Final Model

Random Forest Classifier

```python
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train_scaled,y_train)
```

---

### Performance

| Metric                 | Score  |
| ---------------------- | ------ |
| Test Accuracy          | 93.43% |
| Test Balanced Accuracy | 84.21% |

---

## 💡 Business Value

The model can be used as an Early Warning System (EWS) to identify customers at high risk of churn.

### Benefits

* Reduce churn rate
* Improve retention
* Increase CLV
* Optimize marketing budget

---

# ❓ Question 3: Can Churned Customers Be Segmented For Personalized Promotions?

## 🎯 Objective

The third objective is to segment churned customers into different groups so that the company can design targeted retention campaigns.

---

## 🔍 Analysis Approach

### Step 1: Select Churned Customers

```python
df_churned = df[df['Churn']==1]
```

---

### Step 2: Dimensionality Reduction

PCA was applied before clustering.

```python
pca = PCA(
    n_components=0.90
)

pca_final = pca.fit_transform(
    df_churned_final
)
```

---

### Step 3: K-Means Clustering

```python
kmeans = KMeans(
    n_clusters=4,
    random_state=42
)

clusters = kmeans.fit_predict(
    pca_df
)
```

---

### Step 4: Hierarchical Clustering

```python
agg = AgglomerativeClustering(
    n_clusters=3
)

clusters = agg.fit_predict(
    pca_df
)
```

---

## 📊 Results

### K-Means

* No clear elbow point observed.

### Hierarchical Clustering

* Silhouette Score = 0.132

### Observation

Clusters showed significant overlap and weak separation.

---

## 💡 Conclusion

The available dataset does not provide sufficient information to create meaningful customer segments.

### Possible Reasons

* Limited behavioral variables
* Missing browsing activity
* Missing campaign engagement data

---

## 🚀 Recommendation

Future models should include:

* Website browsing behavior
* Session frequency
* Campaign interaction data
* Customer engagement metrics

These features may improve segmentation quality and support more personalized retention strategies.
