import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from train_model import FEATURE_COLUMNS, PROCESSED_FILE, RANDOM_SEED, TARGET_COLUMN

# Walk-forward folds. Deliberately stop at end of 2024 — 2025-2026 remains
# the untouched final test set (Section 20, #12).
FOLDS = [
    ("Fold 1", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("Fold 2", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("Fold 3", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("Fold 4", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("Fold 5", "2023-12-31", "2024-01-01", "2024-12-31"),
]

THRESHOLDS = np.arange(0.30, 0.71, 0.02)

# Fixed tree count based on prior evidence (early stopping in earlier runs
# consistently landed between 9 and 12 iterations). Not re-tuned per fold —
# that would contaminate each fold's own validation result.
N_ESTIMATORS = 15


def make_model() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=N_ESTIMATORS,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_SEED,
        verbose=-1,
    )


def run_fold(df: pd.DataFrame, train_end: str, val_start: str, val_end: str):
    train = df[df["Date"] <= train_end]
    val = df[(df["Date"] >= val_start) & (df["Date"] <= val_end)]

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_val, y_val = val[FEATURE_COLUMNS], val[TARGET_COLUMN]

    model = make_model()
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)

    return {
        "train_rows": len(train),
        "val_rows": len(val),
        "val_start": val["Date"].min(),
        "val_end": val["Date"].max(),
        "val_prior": y_val.mean(),
        "auc": auc,
        "y_val": y_val.values,
        "val_proba": val_proba,
    }


def threshold_table(y_val: np.ndarray, val_proba: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in THRESHOLDS:
        pred = (val_proba >= t).astype(int)
        n_pos = pred.sum()
        precision = precision_score(y_val, pred, zero_division=0) if n_pos > 0 else np.nan
        rows.append({
            "threshold": round(t, 2),
            "n_pred_pos": n_pos,
            "precision": precision,
            "recall": recall_score(y_val, pred, zero_division=0),
            "f1": f1_score(y_val, pred, zero_division=0),
            "accuracy": accuracy_score(y_val, pred),
        })
    return pd.DataFrame(rows)


def main():
    print(f"Loading {PROCESSED_FILE}")
    df = pd.read_csv(PROCESSED_FILE, parse_dates=["Date"])

    fold_results = []
    fold_threshold_tables = []

    for name, train_end, val_start, val_end in FOLDS:
        result = run_fold(df, train_end, val_start, val_end)
        fold_results.append((name, result))

        print(f"\n=== {name}: train <= {train_end}, val {val_start} to {val_end} ===")
        print(f"Train rows: {result['train_rows']}, Val rows: {result['val_rows']}, "
              f"Val target prior: {result['val_prior']:.3f}")
        print(f"Val AUC: {result['auc']:.3f}")

        tbl = threshold_table(result["y_val"], result["val_proba"])
        fold_threshold_tables.append(tbl.set_index("threshold")["precision"])
        n_pos_tbl = tbl.set_index("threshold")["n_pred_pos"]

        print(tbl.to_string(index=False))

    # --- Aggregation across folds ---
    aucs = [r["auc"] for _, r in fold_results]
    print("\n=== Aggregate across folds ===")
    print(f"AUC per fold: {[round(a, 3) for a in aucs]}")
    print(f"Mean AUC: {np.mean(aucs):.3f}, Std: {np.std(aucs):.3f}, "
          f"Min: {np.min(aucs):.3f}, Max: {np.max(aucs):.3f}")

    precision_by_threshold = pd.concat(fold_threshold_tables, axis=1)
    precision_by_threshold.columns = [name for name, _ in fold_results]
    precision_by_threshold["mean_precision"] = precision_by_threshold.mean(axis=1, skipna=True)
    precision_by_threshold["n_folds_with_predictions"] = precision_by_threshold.drop(
        columns=["mean_precision"]
    ).notna().sum(axis=1)

    print("\nPrecision by threshold, across folds (NaN = no positive predictions that fold):")
    print(precision_by_threshold.round(3).to_string())

    print("\nReminder: 2025-2026 TEST set has not been touched.")


if __name__ == "__main__":
    main()