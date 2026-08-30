import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load processed features
df = pd.read_csv('data/processed/RELIANCE_features.csv', parse_dates=['Date'])

# 13 identical features for 1:1 comparison
features = [
    'return_1d', 'return_5d', 'ma_10', 'ma_20', 'ma_ratio_10_20',
    'volume_change', 'volatility_20', 'rsi_14', 'atr_14', 'momentum_10',
    'nifty_return_1d', 'nifty_return_5d', 'relative_strength_5d'
]
target = 'future_return_5d'

# Strict chronological split: Train (2015-2024) vs Test (2025-2026)
train_df = df[df['Date'] <= '2024-12-31']
test_df = df[df['Date'] > '2024-12-31']

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"Training on {train_df['Date'].min().date()} to {train_df['Date'].max().date()} ({len(X_train)} rows)")
print(f"Testing on  {test_df['Date'].min().date()} to {test_df['Date'].max().date()} ({len(X_test)} rows)")

# Train final Model B v1 using fixed hyperparameters from walk-forward
model = lgb.LGBMRegressor(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    importance_type='gain'
)
model.fit(X_train, y_train)

# Evaluate once on the untouched test set
y_pred = model.predict(X_test)

# --- Metrics Calculation ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Directional Accuracy (ignoring absolute zero actual returns)
valid_dirs = y_test != 0
dir_acc = np.mean(np.sign(y_pred[valid_dirs]) == np.sign(y_test[valid_dirs]))

# Correlation
corr = np.corrcoef(y_test, y_pred)[0, 1]

# --- Naive Baseline Calculation ---
baseline_pred = np.full_like(y_test, y_train.mean())
base_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
base_mae = mean_absolute_error(y_test, baseline_pred)
dominant_sign = np.sign(y_train.mean()) 
base_dir_acc = np.mean(np.sign(y_test[valid_dirs]) == dominant_sign)

print("\n--- FINAL TEST SET EVALUATION (MODEL B) ---")
print(f"RMSE:     {rmse:.4f} (Baseline: {base_rmse:.4f})")
print(f"MAE:      {mae:.4f} (Baseline: {base_mae:.4f})")
print(f"Dir Acc:  {dir_acc:.3f} (Baseline: {base_dir_acc:.3f})")
print(f"Corr:     {corr:.3f}")