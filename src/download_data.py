from pathlib import Path

import pandas as pd
import yfinance as yf


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Initial development instrument
TICKER = "RELIANCE.NS"

# Historical period
START_DATE = "2015-01-01"
END_DATE = "2026-08-27"


def validate_ohlc_sanity(data: pd.DataFrame) -> None:
    """Raw-data sanity checks beyond presence/missingness.

    Raises ValueError with a clear description if any check fails.
    Does not attempt to fix data — only reports problems, since silently
    'fixing' price data can hide real data-quality issues.
    """

    issues = []

    # Date ordering (index should already be datetime, ascending)
    if not data.index.is_monotonic_increasing:
        issues.append("Dates are not in strictly increasing order.")

    # OHLC relationships
    bad_high_low = (data["High"] < data["Low"]).sum()
    if bad_high_low:
        issues.append(f"{bad_high_low} rows where High < Low.")

    bad_high_open = (data["High"] < data["Open"]).sum()
    if bad_high_open:
        issues.append(f"{bad_high_open} rows where High < Open.")

    bad_high_close = (data["High"] < data["Close"]).sum()
    if bad_high_close:
        issues.append(f"{bad_high_close} rows where High < Close.")

    bad_low_open = (data["Low"] > data["Open"]).sum()
    if bad_low_open:
        issues.append(f"{bad_low_open} rows where Low > Open.")

    bad_low_close = (data["Low"] > data["Close"]).sum()
    if bad_low_close:
        issues.append(f"{bad_low_close} rows where Low > Close.")

    # Non-negative volume
    bad_volume = (data["Volume"] < 0).sum()
    if bad_volume:
        issues.append(f"{bad_volume} rows with negative Volume.")

    # Zero volume (not necessarily invalid, but worth flagging, not failing)
    zero_volume = (data["Volume"] == 0).sum()

    # Non-positive prices (invalid for equities)
    for col in ["Open", "High", "Low", "Close"]:
        bad_price = (data[col] <= 0).sum()
        if bad_price:
            issues.append(f"{bad_price} rows with non-positive {col}.")

    if issues:
        detail = "\n  - ".join(issues)
        raise ValueError(f"OHLCV sanity checks failed:\n  - {detail}")

    print("OHLCV sanity checks passed: date ordering, High/Low/Open/Close "
          "relationships, non-negative volume, positive prices.")
    if zero_volume:
        print(f"Note: {zero_volume} rows have zero Volume "
              f"(possible holiday/no-trade rows — not treated as an error).")


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
) -> Path:
    """Download daily historical stock data and save it as CSV."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {ticker}...")
    print(f"Period: {start_date} to {end_date}")

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise RuntimeError(
            f"No data was returned for {ticker}. "
            "Check the ticker symbol or internet connection."
        )

    # Handle yfinance's possible MultiIndex columns.
    if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
        data.columns = data.columns.get_level_values(0)

    data.index.name = "Date"

    # Basic validation (existing)
    if data.index.duplicated().any():
        raise ValueError("Duplicate dates detected in downloaded data.")

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    missing_values = data[required_columns].isna().sum()

    if missing_values.any():
        print("Warning: missing values detected:")
        print(missing_values[missing_values > 0])

    # New: OHLC/volume/date-order sanity checks
    validate_ohlc_sanity(data)

    output_path = RAW_DATA_DIR / "RELIANCE.csv"
    data.to_csv(output_path)

    print()
    print("Download successful.")
    print(f"Rows: {len(data):,}")
    print(f"First date: {data.index.min()}")
    print(f"Last date:  {data.index.max()}")
    print(f"Saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    download_stock_data(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
    )