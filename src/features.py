from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_FILE = RAW_DATA_DIR / "RELIANCE.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "RELIANCE_features.csv"

# Prediction target parameters (Section 8 of handoff)
TARGET_HORIZON_DAYS = 5
TARGET_RETURN_THRESHOLD = 0.01  # >1% gain

# Feature window parameters (Section 7, initial/small set only)
SHORT_MA_WINDOW = 10
LONG_MA_WINDOW = 20
VOLATILITY_WINDOW = 20
RSI_WINDOW = 14
ATR_WINDOW = 14
MOMENTUM_WINDOW = 10


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def remove_non_trading_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove flat, zero-volume placeholder rows.

    These represent dates with no real trading activity (confirmed against
    the NSE holiday calendar that not all such rows correspond to actual
    exchange holidays — some are data-provider artifacts). Detected by
    Volume == 0 combined with Open == High == Low == Close, rather than by
    matching against a holiday calendar, since that's a more robust and
    calendar-independent rule.
    """

    flat_price = (
        (df["Open"] == df["High"])
        & (df["High"] == df["Low"])
        & (df["Low"] == df["Close"])
    )
    zero_volume = df["Volume"] == 0
    non_trading_mask = flat_price & zero_volume

    n_removed = non_trading_mask.sum()
    if n_removed:
        print(f"Removing {n_removed} non-trading placeholder row(s):")
        print(df.loc[non_trading_mask, ["Date", "Volume"]].to_string(index=False))

    return df.loc[~non_trading_mask].reset_index(drop=True)


def compute_rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder's RSI, computed manually (no external 'ta' dependency yet,
    per handoff Section 14 — ta is a later, optional addition)."""

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (equivalent to an EMA with alpha = 1/window)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Where avg_loss is 0 (no losses in window), RSI should be 100
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """Average True Range, computed manually using Wilder's smoothing."""

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return atr


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the initial (small) feature set from handoff Section 7.

    All return/indicator calculations use Adjusted Close, per the project
    decision to avoid split/bonus-issue distortion in raw Close.

    Every feature at row t uses only data available up to and including t.
    """

    out = df.copy()
    adj_close = out["Adj Close"]

    # 1-day and 5-day returns (trailing, feature — not the target)
    out["return_1d"] = adj_close.pct_change(1)
    out["return_5d"] = adj_close.pct_change(5)

    # Moving averages and ratio
    out["ma_10"] = adj_close.rolling(SHORT_MA_WINDOW).mean()
    out["ma_20"] = adj_close.rolling(LONG_MA_WINDOW).mean()
    out["ma_ratio_10_20"] = out["ma_10"] / out["ma_20"]

    # Volume change
    out["volume_change"] = out["Volume"].pct_change(1)

    # 20-day volatility (rolling std of 1-day returns)
    out["volatility_20"] = out["return_1d"].rolling(VOLATILITY_WINDOW).std()

    # RSI(14)
    out["rsi_14"] = compute_rsi(adj_close, RSI_WINDOW)

    # ATR(14) — computed on raw High/Low/Close, since ATR is meant to
    # reflect actual traded price range, not the split-adjusted series
    out["atr_14"] = compute_atr(out["High"], out["Low"], out["Close"], ATR_WINDOW)

    # Momentum (10-day price change)
    out["momentum_10"] = adj_close - adj_close.shift(MOMENTUM_WINDOW)

    # --- Target (Section 8) ---
    # future_close looks FORWARD from t; must never be used as a feature.
    future_close = adj_close.shift(-TARGET_HORIZON_DAYS)
    out["future_return_5d"] = future_close / adj_close - 1
    out["target_up_1pct_5d"] = (
        out["future_return_5d"] > TARGET_RETURN_THRESHOLD
    ).astype(int)

    return out


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that lack full feature history (warm-up) or a valid
    forward-looking target (trailing). These rows aren't usable for
    training/evaluation and are dropped explicitly rather than filled."""

    feature_cols = [
        "return_1d",
        "return_5d",
        "ma_10",
        "ma_20",
        "ma_ratio_10_20",
        "volume_change",
        "volatility_20",
        "rsi_14",
        "atr_14",
        "momentum_10",
    ]
    target_cols = ["future_return_5d", "target_up_1pct_5d"]

    before = len(df)
    df = df.dropna(subset=feature_cols + ["future_return_5d"]).reset_index(drop=True)
    after = len(df)

    print(f"Dropped {before - after} row(s) lacking full feature warm-up "
          f"or a valid forward target ({before} -> {after} rows).")

    return df


def main() -> Path:
    print(f"Loading raw data from {RAW_FILE}")
    df = load_raw_data(RAW_FILE)
    print(f"Loaded {len(df)} raw rows.")

    df = remove_non_trading_rows(df)
    print(f"{len(df)} rows remain after removing non-trading placeholder rows.")

    df = build_features(df)
    df = finalize_dataset(df)

    # Sanity check: target should not be perfectly correlated with any
    # single feature (a red flag for leakage if it were).
    print()
    print("Target class balance (target_up_1pct_5d):")
    print(df["target_up_1pct_5d"].value_counts(normalize=True).round(3))

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print(f"Saved processed feature set to: {OUTPUT_FILE}")
    print(f"Final shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    return OUTPUT_FILE


if __name__ == "__main__":
    main()