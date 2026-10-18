
from pathlib import Path

import numpy as np
import pandas as pd

# Reproducible synthetic financial data
rng = np.random.default_rng(42)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "sample_data" / "sample_companies.xlsx"

dates = pd.bdate_range("2023-01-02", "2023-12-29")

companies = [
    ("Demo Manufacturing Bhd", 1_500_000_000, 400_000_000, 150_000_000),
    ("Demo Technology Bhd", 950_000_000, 180_000_000, 80_000_000),
    ("Demo Holdings Bhd", 2_200_000_000, 650_000_000, 220_000_000),
]

rows = []

for name, initial_equity, long_debt, short_debt in companies:
    daily_returns = rng.normal(0.0002, 0.015, len(dates))
    equity_values = initial_equity * np.exp(np.cumsum(daily_returns))

    for date, equity in zip(dates, equity_values):
        rows.append({
            "DATE": date,
            "COMPANY NAME": name,
            "MARKET EQUITY VALUE": round(float(equity), 2),
            "LONG TERM DEBT": long_debt,
            "SHORT TERM DEBT": short_debt,
            "RISK FREE RATE (%)": 0.035,
        })

df = pd.DataFrame(rows)
df = df.sort_values(["COMPANY NAME", "DATE"])

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_excel(OUTPUT_FILE, index=False)

print(f"Created: {OUTPUT_FILE}")
print(f"Total observations: {len(df)}")
