# CODSOFT_TASK2 — Credit Card Fraud Detection

Detects fraudulent credit card transactions using engineered features and classic machine learning classifiers.

## 📌 Problem Statement

Given a credit card transaction's details (amount, category, time, customer demographics), predict whether it is fraudulent or legitimate — a highly imbalanced binary classification problem (fraud is only ~0.6% of transactions).

## 📂 Dataset

[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Kaggle)

- `fraudTrain.csv` — 1,296,675 labeled transactions
- `fraudTest.csv` — 555,719 labeled transactions used for evaluation

## 🛠️ Approach

1. **Exploratory Data Analysis**
   - Confirmed severe class imbalance (99.42% legitimate vs. 0.58% fraud)
   - Identified key fraud signals through evidence-based analysis:
     - Transaction amount (fraud averages ~8x higher: $531 vs. $68)
     - Purchase category (online categories like `shopping_net` had 3x+ higher fraud rates than in-person categories)
     - Time of day (late-night transactions, 10pm–3am, had ~18x higher fraud rate than daytime)
   - Ruled out geographic distance between cardholder and merchant — showed no meaningful difference between fraud and legitimate transactions

2. **Feature Engineering**
   - Extracted `is_night` (binary flag for 10pm–3am transactions)
   - Extracted customer `age` from date of birth
   - One-hot encoded `category` and `gender`
   - Dropped identifying/noise columns (names, card numbers, street address, transaction IDs)
   - Scaled numeric features with `StandardScaler` (fit on train, applied to test only)

3. **Model Training**
   Trained and compared three classifiers, all using `class_weight='balanced'` to address the imbalance:
   - Logistic Regression
   - Decision Tree
   - Random Forest

4. **Evaluation**
   Evaluated using precision, recall, F1-score, confusion matrix, and ROC-AUC for the fraud class specifically — accuracy alone is misleading here, since predicting "no fraud" for every transaction would already score 99.4% accuracy while catching zero fraud.

## 📊 Results

| Model | Precision (fraud) | Recall (fraud) | F1-score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.02 | 0.89 | 0.05 | 0.947 |
| Decision Tree | 0.09 | 0.97 | 0.16 | 0.986 |
| Random Forest | **0.13** | 0.95 | **0.23** | **0.992** ✅ (best) |

Random Forest performed best overall — catching 95% of fraud cases (2,035 of 2,145) with the best precision and lowest false-positive rate (13,765) among the three, backed by a strong 0.992 ROC-AUC. Decision Tree had marginally higher raw recall but produced far more false alarms (22,170), making Random Forest the better practical trade-off.

## 📁 Repository Contents

- `CODSOFT_TASK2_Credit_Card_Fraud_Detection.ipynb` — full notebook (EDA, feature engineering, model training, evaluation)
- `README.md` — this file

## ▶️ How to Run

1. Open the notebook in Google Colab or Jupyter
2. Upload `fraudTrain.csv` and `fraudTest.csv`
3. Run all cells in order

## 🔮 Possible Improvements

- Threshold tuning to improve precision without sacrificing recall
- Resampling techniques (SMOTE) as an alternative to `class_weight='balanced'`
- Try Gradient Boosting / XGBoost for potentially stronger performance
