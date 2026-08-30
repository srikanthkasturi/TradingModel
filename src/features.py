from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_FILE = RAW_DATA_DIR / "RELIANCE.csv"
NIFTY_RAW_FILE = RAW_DATA_DIR / "NIFTY.csv"
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


def load_nifty_close(path: Path) -> pd.DataFrame:
    """Load only the Nifty Adj Close, renamed to avoid column collisions
    when merged with the RELIANCE dataframe."""

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[["Date", "Adj Close"]].rename(columns={"Adj Close": "nifty_adj_close"})
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def merge_with_nifty(reliance_df: pd.DataFrame, nifty_df: pd.DataFrame) -> pd.DataFrame:
    """Inner join on Date. Reports any dates present in one but not the
    other, rather than silently dropping them."""

    reliance_dates = set(reliance_df["Date"])
    nifty_dates = set(nifty_df["Date"])

    only_reliance = reliance_dates - nifty_dates
    only_nifty = nifty_dates - reliance_dates

    if only_reliance:
        print(f"{len(only_reliance)} RELIANCE date(s) have no matching NIFTY "
              f"row and will be dropped from the merged dataset.")
    if only_nifty:
        print(f"{len(only_nifty)} NIFTY date(s) have no matching RELIANCE "
              f"row and will be ignored.")

    merged = pd.merge(reliance_df, nifty_df, on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged


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
    out = df.copy()
    adj_close = out["Adj Close"]

    # --- BASE 13 FEATURES ---
    out["return_1d"] = adj_close.pct_change(1)
    out["return_5d"] = adj_close.pct_change(5)
    out["ma_10"] = adj_close.rolling(SHORT_MA_WINDOW).mean()
    out["ma_20"] = adj_close.rolling(LONG_MA_WINDOW).mean()
    out["ma_ratio_10_20"] = out["ma_10"] / out["ma_20"]
    out["volume_change"] = out["Volume"].pct_change(1)
    out["volatility_20"] = out["return_1d"].rolling(VOLATILITY_WINDOW).std()
    out["rsi_14"] = compute_rsi(adj_close, RSI_WINDOW)
    out["atr_14"] = compute_atr(out["High"], out["Low"], out["Close"], ATR_WINDOW)
    out["momentum_10"] = adj_close - adj_close.shift(MOMENTUM_WINDOW)
    
    nifty_close = out["nifty_adj_close"]
    out["nifty_return_1d"] = nifty_close.pct_change(1)
    out["nifty_return_5d"] = nifty_close.pct_change(5)
    out["relative_strength_5d"] = out["return_5d"] - out["nifty_return_5d"]

    # --- MODEL C v2: VOLUME DYNAMICS FEATURES ---
    
    # 1. Volume MA Ratio (5, 20)
    vol_ma_5 = out["Volume"].rolling(window=5).mean()
    vol_ma_20 = out["Volume"].rolling(window=20).mean()
    out["volume_ma_ratio_5_20"] = vol_ma_5 / vol_ma_20
    
    # 2. On-Balance Volume (OBV)
    # Use np.sign of daily return to determine volume direction (+1, -1, 0)
    price_change_sign = np.sign(out["return_1d"])
    signed_volume = out["Volume"] * price_change_sign
    out["obv"] = signed_volume.cumsum()
    
    # 3. OBV Slope (10-day normalized trend)
    vol_ma_10 = out["Volume"].rolling(window=10).mean()
    out["obv_slope_10"] = (out["obv"] - out["obv"].shift(10)) / vol_ma_10

    # --- Target (Section 8) ---
    future_close = adj_close.shift(-TARGET_HORIZON_DAYS)
    out["future_return_5d"] = future_close / adj_close - 1
    out["target_up_1pct_5d"] = (out["future_return_5d"] > TARGET_RETURN_THRESHOLD).astype(int)

    return out


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "return_1d", "return_5d", "ma_10", "ma_20", "ma_ratio_10_20",
        "volume_change", "volatility_20", "rsi_14", "atr_14", "momentum_10",
        "nifty_return_1d", "nifty_return_5d", "relative_strength_5d",
        "volume_ma_ratio_5_20", "obv", "obv_slope_10"
    ]
    
    before = len(df)
    df = df.dropna(subset=feature_cols + ["future_return_5d"]).reset_index(drop=True)
    after = len(df)

    print(f"Dropped {before - after} row(s) lacking full feature warm-up or a valid forward target ({before} -> {after} rows).")
    return df


def main() -> Path:
    print(f"Loading raw RELIANCE data from {RAW_FILE}")
    df = load_raw_data(RAW_FILE)
    print(f"Loaded {len(df)} raw RELIANCE rows.")

    df = remove_non_trading_rows(df)
    print(f"{len(df)} RELIANCE rows remain after removing non-trading placeholder rows.")

    print(f"\nLoading NIFTY data from {NIFTY_RAW_FILE}")
    nifty_df = load_nifty_close(NIFTY_RAW_FILE)
    print(f"Loaded {len(nifty_df)} NIFTY rows.")
    # Note: NIFTY's zero-volume rows are NOT filtered here — index volume
    # is not a meaningful signal, unlike for RELIANCE.

    df = merge_with_nifty(df, nifty_df)
    print(f"{len(df)} rows remain after merging with NIFTY on Date.")

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