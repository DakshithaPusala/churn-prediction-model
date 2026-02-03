import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

print("=" * 60)
print("CUSTOMER CHURN PREDICTION MODEL")
print("=" * 60)

# ==================== PHASE 1: LOAD DATA ====================
print("\n[1/6] Loading data...")
try:
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    print(f"✓ Dataset loaded successfully!")
    print(f"  Shape: {df.shape[0]} customers, {df.shape[1]} features")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit()

# Quick exploration
print(f"\n  Churn breakdown:")
print(f"  - Customers who stayed: {(df['Churn'] == 'No').sum()}")
print(f"  - Customers who left: {(df['Churn'] == 'Yes').sum()}")
print(f"  - Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")

# ==================== PHASE 2: PREPARE DATA ====================
print("\n[2/6] Preparing data...")

# Drop unnecessary columns
df_clean = df.drop(['customerID'], axis=1)

# Convert target variable to numbers (Yes=1, No=0)
df_clean['Churn'] = (df_clean['Churn'] == 'Yes').astype(int)

# Find categorical columns (text columns)
categorical_cols = [col for col in df_clean.columns if df_clean[col].dtype == 'object']

# Convert text to numbers using LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le
    print(f"  ✓ Encoded: {col}")

# Separate features (X) and target (y)
X = df_clean.drop('Churn', axis=1)
y = df_clean['Churn']

print(f"\n  ✓ Data prepared!")
print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (normalize values)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  - Training samples: {X_train.shape[0]}")
print(f"  - Testing samples: {X_test.shape[0]}")

# ==================== PHASE 3: TRAIN MODELS ====================
print("\n[3/6] Training machine learning models...")

# Model 1: Logistic Regression (fast, interpretable)
print("  Training: Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
print("  ✓ Logistic Regression trained")

# Model 2: Decision Tree (captures complex patterns)
print("  Training: Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42, min_samples_split=20)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("  ✓ Decision Tree trained")

# ==================== PHASE 4: EVALUATE MODELS ====================
print("\n[4/6] Evaluating model performance...")

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and display metrics"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n  {model_name}:")
    print(f"  - Accuracy:  {acc:.1%} (overall correctness)")
    print(f"  - Precision: {prec:.1%} (correct when predicting churn)")
    print(f"  - Recall:    {rec:.1%} (catches actual churners)")
    print(f"  - F1 Score:  {f1:.3f} (overall balance)")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

lr_metrics = evaluate_model(y_test, lr_pred, "Logistic Regression")
dt_metrics = evaluate_model(y_test, dt_pred, "Decision Tree")

# ==================== PHASE 5: SELECT BEST MODEL ====================
print("\n[5/6] Selecting best model...")

if lr_metrics['f1'] > dt_metrics['f1']:
    best_model = lr_model
    best_name = "Logistic Regression"
    print(f"  ✓ Selected: {best_name}")
else:
    best_model = dt_model
    best_name = "Decision Tree"
    print(f"  ✓ Selected: {best_name}")

# Create models directory
os.makedirs('models', exist_ok=True)

# Save model
with open('models/churn_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"  ✓ Model saved to: models/churn_model.pkl")

# ==================== PHASE 6: CREATE VISUALIZATIONS ====================
print("\n[6/6] Creating visualizations...")

os.makedirs('visualizations', exist_ok=True)

# Chart 1: Churn distribution
plt.figure(figsize=(8, 5))
churn_counts = df['Churn'].value_counts()
colors = ['#2ecc71', '#e74c3c']  # Green for No, Red for Yes
plt.bar(['No', 'Yes'], [churn_counts[0], churn_counts[1]], color=colors, width=0.6)
plt.title('Customer Churn Distribution', fontsize=14, fontweight='bold')
plt.ylabel('Number of Customers', fontsize=12)
plt.xlabel('Churn Status', fontsize=12)
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate([churn_counts[0], churn_counts[1]]):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/churn_distribution.png', dpi=300)
plt.close()
print("  ✓ Saved: churn_distribution.png")

# Chart 2: Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='#3498db')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 10 Features Predicting Churn', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=300)
plt.close()
print("  ✓ Saved: feature_importance.png")

# Chart 3: Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, lr_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'],
            annot_kws={'size': 14, 'weight': 'bold'})
plt.title('Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix.png', dpi=300)
plt.close()
print("  ✓ Saved: confusion_matrix.png")

# ==================== SUMMARY ====================
print("\n" + "=" * 60)
print("✓ PROJECT COMPLETE!")
print("=" * 60)
print(f"\nResults:")
print(f"  - Model trained successfully")
print(f"  - Accuracy: {lr_metrics['accuracy']:.1%}")
print(f"  - Can identify {lr_metrics['recall']:.0%} of customers who will churn")
print(f"  - Model saved: models/churn_model.pkl")
print(f"  - Visualizations saved: visualizations/")
print(f"\nNext step: Upload to GitHub!")
print("=" * 60)
