import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data exported from MySQL
df = pd.read_csv('data/customer_churn_clean.csv')

# Basic checks
print(df.shape)
print(df.info())
print(df.head())

# Set consistent style for all plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

# 1. Churn distribution (visual version of our SQL baseline query)
plt.figure()
ax = sns.countplot(data=df, x='Churn', hue='Churn', palette=['#2ecc71', '#e74c3c'], legend=False)
plt.title('Customer Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')

# Add percentage labels on top of bars
total = len(df)
for p in ax.patches:
    pct = f'{100 * p.get_height() / total:.1f}%'
    ax.annotate(pct, (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('charts/01_churn_distribution.png', dpi=150)
plt.show()

# 2. Churn rate by Contract type
contract_churn = df.groupby('Contract')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index(name='churn_rate')
contract_churn = contract_churn.sort_values('churn_rate', ascending=False)

plt.figure()
ax = sns.barplot(data=contract_churn, x='Contract', y='churn_rate', hue='Contract',
                  palette='Reds_r', legend=False)
plt.title('Churn Rate by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Churn Rate (%)')

for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', 
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('charts/02_churn_by_contract.png', dpi=150)
plt.show()

# 3. Correlation heatmap
# Encode categorical columns to numeric for correlation analysis
df_encoded = df.copy()
df_encoded['Churn_numeric'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Select relevant categorical + numeric columns to encode
cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod']

for col in cat_cols:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

# Build correlation matrix against Churn_numeric
corr_cols = cat_cols + ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_numeric']
corr_matrix = df_encoded[corr_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            linewidths=0.5, annot_kws={'size': 8})
plt.title('Correlation Heatmap - All Features vs Churn')
plt.tight_layout()
plt.savefig('charts/03_correlation_heatmap.png', dpi=150)
plt.show()

from scipy.stats import chi2_contingency

# Chi-square test: Contract vs Churn
contingency_table = pd.crosstab(df['Contract'], df['Churn'])
print("Contingency Table:")
print(contingency_table)

chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square statistic: {chi2:.2f}")
print(f"P-value: {p_value:.10f}")
print(f"Degrees of freedom: {dof}")

if p_value < 0.05:
    print("Result: Statistically significant relationship between Contract and Churn (p < 0.05)")
else:
    print("Result: No statistically significant relationship found")
from scipy.stats import ttest_ind

churned = df[df['Churn'] == 'Yes']['MonthlyCharges']
retained = df[df['Churn'] == 'No']['MonthlyCharges']

t_stat, p_val = ttest_ind(churned, retained, equal_var=False)

print(f"Churned mean: {churned.mean():.2f}")
print(f"Retained mean: {retained.mean():.2f}")
print(f"T-statistic: {t_stat:.2f}")
print(f"P-value: {p_val:.10f}")

if p_val < 0.05:
    print("Result: Statistically significant difference in MonthlyCharges between churned and retained customers")
else:
    print("Result: No statistically significant difference found")
    
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Drop customerID (not a predictive feature) and encode target
model_df = df.drop(columns=['customerID']).copy()
model_df['Churn'] = model_df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encode all categorical columns (better than .cat.codes for modeling —
# avoids implying false ordinal relationships between categories)
model_df = pd.get_dummies(model_df, drop_first=True)

# Separate features (X) and target (y)
X = model_df.drop(columns=['Churn'])
y = model_df['Churn']

print("Feature matrix shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# Train-test split — stratify=y keeps the same churn ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features (tenure, MonthlyCharges, TotalCharges) — 
# Logistic Regression is sensitive to feature scale
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

from sklearn.linear_model import LogisticRegression

# Baseline model — no class balancing yet, so we can see the imbalance problem firsthand
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print("Baseline model trained.")
print("Predicted churn count in test set:", (y_pred == 1).sum())
print("Actual churn count in test set:", (y_test == 1).sum())

from sklearn.metrics import (confusion_matrix, classification_report, 
                               roc_auc_score, roc_curve, ConfusionMatrixDisplay)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Baseline Logistic Regression')
plt.tight_layout()
plt.savefig('charts/04_confusion_matrix_baseline.png', dpi=150)
plt.show()

# Precision, recall, F1
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# ROC-AUC
y_proba = log_reg.predict_proba(X_test)[:, 1]  # probability of churn (class 1)
auc_score = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {auc_score:.4f}")

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.3f})', color='#e74c3c')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Baseline Model')
plt.legend()
plt.tight_layout()
plt.savefig('charts/05_roc_curve_baseline.png', dpi=150)
plt.show()


# Retrain with class balancing — penalizes the model more for missing churn cases
log_reg_balanced = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log_reg_balanced.fit(X_train, y_train)

y_pred_balanced = log_reg_balanced.predict(X_test)
y_proba_balanced = log_reg_balanced.predict_proba(X_test)[:, 1]

print("Balanced Model Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_balanced))

print("\nBalanced Model Classification Report:")
print(classification_report(y_test, y_pred_balanced, target_names=['No Churn', 'Churn']))

auc_balanced = roc_auc_score(y_test, y_proba_balanced)
print(f"Balanced Model ROC-AUC: {auc_balanced:.4f}")

# Feature importance from Logistic Regression coefficients
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': log_reg_balanced.coef_[0]
})
feature_importance['abs_coefficient'] = feature_importance['coefficient'].abs()
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

print("Top 10 features driving churn prediction:")
print(feature_importance.head(10)[['feature', 'coefficient']])

# Visualize top 10
plt.figure(figsize=(10, 6))
top10 = feature_importance.head(10)
colors = ['#e74c3c' if c > 0 else '#3498db' for c in top10['coefficient']]
plt.barh(top10['feature'], top10['coefficient'], color=colors)
plt.xlabel('Coefficient (Red = increases churn risk, Blue = decreases churn risk)')
plt.title('Top 10 Features Predicting Churn')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('charts/06_feature_importance.png', dpi=150)
plt.show()