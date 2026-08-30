
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import numpy as np

# Load processed features
df = pd.read_csv('data/processed/RELIANCE_features.csv', parse_dates=['Date'])

# Define final feature set and target
features = [
    'return_1d', 'return_5d', 'ma_10', 'ma_20', 'ma_ratio_10_20',
    'volume_change', 'volatility_20', 'rsi_14', 'atr_14', 'momentum_10',
    'nifty_return_1d', 'nifty_return_5d', 'relative_strength_5d'
]
target = 'target_up_1pct_5d'

# Strict chronological split: Train (2015-2024) vs Test (2025-2026)
train_df = df[df['Date'] <= '2024-12-31']
test_df = df[df['Date'] > '2024-12-31']

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"Training on {train_df['Date'].min().date()} to {train_df['Date'].max().date()} ({len(X_train)} rows)")
print(f"Testing on  {test_df['Date'].min().date()} to {test_df['Date'].max().date()} ({len(X_test)} rows)")
print(f"Test target prior: {y_test.mean():.3f}\n")

# Train final Model A v1 using fixed iterations from earlier early-stopping runs (e.g., 12 trees)
model = lgb.LGBMClassifier(
    n_estimators=12,
    random_state=42,
    importance_type='gain'
)
model.fit(X_train, y_train)

# Evaluate once on the untouched test set
y_test_pred_prob = model.predict_proba(X_test)[:, 1]

print("--- FINAL TEST SET EVALUATION ---")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_test_pred_prob):.3f}")

print("\n--- Threshold sweep (TEST) ---")
print(f"Predicted probability range: [{y_test_pred_prob.min():.3f}, {y_test_pred_prob.max():.3f}], mean = {y_test_pred_prob.mean():.3f}")
print(" thresh n_pred_pos  precision   recall       f1  accuracy")

thresholds = np.arange(0.30, 0.72, 0.02)
for t in thresholds:
    preds = (y_test_pred_prob >= t).astype(int)
    n_pred = preds.sum()
    if n_pred > 0:
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
    else:
        prec, rec, f1 = np.nan, 0.0, 0.0
    acc = accuracy_score(y_test, preds)
    
    print(f"  {t:.2f}         {n_pred:3d}      {prec:.3f}    {rec:.3f}    {f1:.3f}     {acc:.3f}")