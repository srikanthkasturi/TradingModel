import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('data/processed/RELIANCE_features.csv', parse_dates=['Date'])

# 13 Base + 3 New Volume Dynamics features (16 total)
features = [
    'return_1d', 'return_5d', 'ma_10', 'ma_20', 'ma_ratio_10_20',
    'volume_change', 'volatility_20', 'rsi_14', 'atr_14', 'momentum_10',
    'nifty_return_1d', 'nifty_return_5d', 'relative_strength_5d',
    'volume_ma_ratio_5_20', 'obv', 'obv_slope_10'
]
target = 'future_return_5d'

folds = [
    {'train_end': '2019-12-31', 'val_start': '2020-01-01', 'val_end': '2020-12-31'},
    {'train_end': '2020-12-31', 'val_start': '2021-01-01', 'val_end': '2021-12-31'},
    {'train_end': '2021-12-31', 'val_start': '2022-01-01', 'val_end': '2022-12-31'},
    {'train_end': '2022-12-31', 'val_start': '2023-01-01', 'val_end': '2023-12-31'},
    {'train_end': '2023-12-31', 'val_start': '2024-01-01', 'val_end': '2024-12-31'}
]

print("=== Model C v2 (Volume Dynamics) Walk-Forward Validation ===")

metrics_list = []
feature_importances = np.zeros(len(features))

for i, fold in enumerate(folds, 1):
    train_mask = df['Date'] <= fold['train_end']
    val_mask = (df['Date'] >= fold['val_start']) & (df['Date'] <= fold['val_end'])
    
    X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, target]
    X_val, y_val = df.loc[val_mask, features], df.loc[val_mask, target]
    
    model = lgb.LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.05, random_state=42, importance_type='gain')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    feature_importances += model.feature_importances_
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    valid_dirs = y_val != 0
    dir_acc = np.mean(np.sign(y_pred[valid_dirs]) == np.sign(y_val[valid_dirs]))
    corr = np.corrcoef(y_val, y_pred)[0, 1]
    
    baseline_pred = np.full_like(y_val, y_train.mean())
    base_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
    base_mae = mean_absolute_error(y_val, baseline_pred)
    base_dir_acc = np.mean(np.sign(y_val[valid_dirs]) == np.sign(y_train.mean()))

    metrics_list.append({
        'fold': i, 'rmse': rmse, 'base_rmse': base_rmse, 'mae': mae, 'base_mae': base_mae,
        'dir_acc': dir_acc, 'base_dir_acc': base_dir_acc, 'corr': corr
    })
    
    print(f"\n--- Fold {i} ({fold['val_start'][:4]}) ---")
    print(f"RMSE:     {rmse:.4f} (Base: {base_rmse:.4f})")
    print(f"Dir Acc:  {dir_acc:.3f} (Base: {base_dir_acc:.3f})")
    print(f"Corr:     {corr:.3f}")

print("\n=== Aggregate Across Folds ===")
agg_df = pd.DataFrame(metrics_list)
print(f"Mean RMSE:    {agg_df['rmse'].mean():.4f} (Base: {agg_df['base_rmse'].mean():.4f} | Model B v1: 0.0406)")
print(f"Mean MAE:     {agg_df['mae'].mean():.4f} (Base: {agg_df['base_mae'].mean():.4f} | Model B v1: 0.0310)")
print(f"Mean Dir Acc: {agg_df['dir_acc'].mean():.3f} (Base: {agg_df['base_dir_acc'].mean():.3f} | Model B v1: 0.519)")
print(f"Mean Corr:    {agg_df['corr'].mean():.3f} (Model B v1: -0.014)")

print("\n=== Feature Importances (Gain across all folds) ===")
importance_df = pd.DataFrame({'Feature': features, 'Gain': feature_importances}).sort_values(by='Gain', ascending=False)
for _, row in importance_df.iterrows():
    print(f"{row['Feature']:<22} {row['Gain']:.1f}")

print("\nReminder: 2025-2026 TEST set has not been touched.")
