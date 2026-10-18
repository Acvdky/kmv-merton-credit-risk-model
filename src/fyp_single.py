import warnings
warnings.filterwarnings("ignore", message="The iteration is not making good progress", category=RuntimeWarning)

from pathlib import Path
from math import log, sqrt, exp
from typing import Optional, Tuple, Iterable

import logging
import numpy as np
import pandas as pd

# Force pandas to display floats in scientific notation uppercase E with 5 significant digits
pd.set_option('display.float_format', lambda x: f"{x:.5E}")

from scipy.stats import norm
from scipy.optimize import fsolve
from decimal import Decimal, getcontext, ROUND_HALF_UP

# --------------------- Config ---------------------
EXCEL_PATH: Path = Path("IHH HEALTHCARE DATA 2014.xlsm")
SHEET_NAME: Optional[str] = None
MAX_ROWS: Optional[int] = 200
OUTPUT_DIR: Path = Path.cwd() / "Outputs"

# Market/Model parameters
T: float = 1.0
TRADING_DAYS_PER_YEAR: int = 200

# Numerical constants
SE_FALLBACK: float = 0.3
XTOL: float = 1e-9
MAXFEV: int = 2000
SV_CAP: float = 3.0

# Decimal precision context for quantizing final value to 9 decimal places
getcontext().prec = 28

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --------------------- Utility functions ---------------------
def to_f(x) -> float:
    """
    Parse cell-like inputs to floats.
    Returns:
      - decimal numeric value (e.g. "3%" -> 0.03, "$1,200" -> 1200.0)
      - np.nan on blank/parse error
    """
    try:
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s == "":
            return np.nan
        if s.endswith("%"):
            try:
                return float(s[:-1].strip()) / 100.0
            except Exception:
                return np.nan
        # remove known currency symbols and commas
        for sym in ("$", "RM", "USD", "SGD", "MYR"):
            s = s.replace(sym, "")
        s = s.replace(",", "").strip()
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        if s == "":
            return np.nan
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def get_log_rf_from_decimal(r_decimal: float) -> float:
    """
    Compute log risk-free rate = ln(1 + r_decimal) and return rounded to 9 decimal places.
    r_decimal: expected as decimal (e.g. 0.0301 for 3.01%).
    If r_decimal is NaN, returns NaN.
    """
    if pd.isna(r_decimal):
        return np.nan
    try:
        raw = log(1.0 + float(r_decimal))
        d = Decimal(str(raw))
        quantized = d.quantize(Decimal("0.000000001"), rounding=ROUND_HALF_UP)
        return float(quantized)
    except Exception as exc:
        logger.debug("get_log_rf_from_decimal error: %s (input=%s)", exc, r_decimal)
        return np.nan


def compute_d1(V: float, D: float, sigma: float, r_log: float, T_val: float = 1.0) -> float:
    """
    Standard d1 used in Merton/Black-Scholes formulas.
    Returns NaN on invalid inputs to avoid exceptions propagation.
    """
    try:
        if any(pd.isna(x) for x in (V, D, sigma, r_log, T_val)):
            return np.nan
        if sigma <= 0 or D <= 0 or V <= 0:
            return np.nan
        return (log(V / D) + (r_log + 0.5 * sigma * sigma) * T_val) / (sigma * sqrt(T_val))
    except Exception as exc:
        logger.debug("compute_d1 error: %s (V=%s, D=%s, sigma=%s, r_log=%s)", exc, V, D, sigma, r_log)
        return np.nan


# --------------------- KMV solver ---------------------
def kmv_solve(
    E: float,
    D: float,
    r_log: float,
    T_val: float,
    se: float,
    v0: Optional[float] = None,
    sv0: Optional[float] = None,
    maxfev: int = MAXFEV,
) -> Tuple[float, float]:
    """
    Solve system:
      eq1: V * N(d1) - D * exp(-r_log * T) * N(d2) - E = 0
      eq2: sV * V * N(d1) - se * E = 0

    Returns (V_sol, sV_sol) or (np.nan, np.nan) on failure.
    This function keeps the same formulas and constraints as your original code.
    """
    # Basic input validation
    try:
        if pd.isna(r_log):
            return np.nan, np.nan
        if not (E > 0 and D > 0 and T_val > 0 and se > 0):
            return np.nan, np.nan
    except Exception:
        return np.nan, np.nan

    V0 = float(E if v0 is None else v0)
    sV0 = float(se if sv0 is None else sv0)

    def equations(vars_: Iterable[float]) -> Iterable[float]:
        V, sV = vars_
        # enforce minimal positive values internally to avoid log/zero issues
        if V <= 0:
            V = 1e-9
        if sV <= 0:
            sV = 1e-9
        if sV > SV_CAP:
            sV = SV_CAP
        svt = sV * sqrt(T_val)
        # compute d1 and d2 with try to catch domain errors
        try:
            d1 = (log(V / D) + (r_log + 0.5 * sV * sV) * T_val) / svt
        except Exception:
            # return large residuals so solver steers away
            return [1e9, 1e9]
        d2 = d1 - svt
        n1 = norm.cdf(d1)
        n2 = norm.cdf(d2)
        eq1 = V * n1 - D * exp(-r_log * T_val) * n2 - E
        eq2 = sV * V * n1 - se * E
        return [eq1, eq2]

    try:
        sol, info, ier, msg = fsolve(equations, [V0, sV0], xtol=XTOL, maxfev=maxfev, full_output=True)
    except Exception as exc:
        logger.debug("kmv_solve fsolve exception: %s (E=%s, D=%s, r_log=%s)", exc, E, D, r_log)
        return np.nan, np.nan

    # fsolve success flag is ier == 1
    if ier != 1:
        logger.debug("kmv_solve fsolve failed: ier=%s, msg=%s, inputs=(E=%s, D=%s, r_log=%s)", ier, msg, E, D, r_log)
        return np.nan, np.nan

    try:
        V_sol = float(sol[0])
        sV_sol = float(sol[1])
        if V_sol <= 0 or sV_sol <= 0:
            return np.nan, np.nan
        return V_sol, sV_sol
    except Exception as exc:
        logger.debug("kmv_solve solution parsing error: %s (sol=%s)", exc, sol)
        return np.nan, np.nan


# --------------------- I/O & Data preparation ---------------------
def load_and_prepare_dataframe(excel_path: Path, sheet_name: Optional[str], max_rows: Optional[int]) -> pd.DataFrame:
    """
    Load excel and map columns to required fields. Returns prepared DataFrame with:
      DATE, MARKET EQUITY VALUE, LONG TERM DEBT, SHORT TERM DEBT, RISK FREE RATE (%), DPT,
      LOG RISK FREE RATE, TIME HORIZON
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"{excel_path} not found.")

    xls = pd.ExcelFile(excel_path)
    sheet = sheet_name or (xls.sheet_names[0] if len(xls.sheet_names) == 1 else None)
    if sheet and sheet not in xls.sheet_names:
        raise ValueError(f"Sheet '{sheet}' not found. Available: {xls.sheet_names}")

    read_kwargs = {}
    if max_rows is not None:
        read_kwargs["nrows"] = max_rows
    df_raw = pd.read_excel(xls, sheet_name=sheet, **read_kwargs)

    # build case-insensitive column map
    colmap = {c.strip().upper(): c for c in df_raw.columns}

    def find_col(keys):
        for k in keys:
            k_up = k.strip().upper()
            for up, orig in colmap.items():
                if k_up in up:
                    return orig
        return None

    col_date = find_col(["DATE"])
    col_equity = find_col(["MARKET EQUITY", "MARKET VALUE", "EQUITY"])
    col_ltd = find_col(["LONG TERM", "LONG_TERM", "LTD"])
    col_std = find_col(["SHORT TERM", "SHORT_TERM", "STD"])
    col_risk = find_col(["RISK FREE", "RISK_FREE", "RISK"])

    if not all([col_date, col_equity, col_ltd, col_std]):
        raise ValueError("Required columns not found. Expected fields: DATE, MARKET EQUITY, LONG TERM, SHORT TERM")

    df = pd.DataFrame()
    df["DATE"] = pd.to_datetime(df_raw[col_date], errors="coerce")
    df["MARKET EQUITY VALUE"] = df_raw[col_equity].apply(to_f)
    df["LONG TERM DEBT"] = df_raw[col_ltd].apply(to_f).fillna(0.0)
    df["SHORT TERM DEBT"] = df_raw[col_std].apply(to_f).fillna(0.0)
    df["RISK FREE RATE (%)"] = df_raw[col_risk].apply(to_f) if col_risk else np.nan
    df["DPT"] = 0.5 * df["LONG TERM DEBT"] + df["SHORT TERM DEBT"]

    # compute and store LOG RISK FREE RATE using original quantization rule
    df["LOG RISK FREE RATE"] = df["RISK FREE RATE (%)"].apply(get_log_rf_from_decimal)
    df["TIME HORIZON"] = T

    return df


# --------------------- Metrics computation ---------------------
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all metrics and preserve original formulas/behaviour.
    Returns df with computed columns (mutates a copy).
    """
    df = df.copy()

    df["INITIAL MV OF ASSET"] = df["MARKET EQUITY VALUE"]

    series_init = df["INITIAL MV OF ASSET"].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    mv_of_asset_return = series_init.pct_change()
    if len(mv_of_asset_return) > 0:
        mv_of_asset_return.iloc[0] = np.nan
    df["MV OF ASSET RETURN"] = mv_of_asset_return

    sigma_v_raw = np.sqrt(TRADING_DAYS_PER_YEAR) * df["MV OF ASSET RETURN"].dropna().std(ddof=1)
    df["INITIAL MV OF ASSET VOLATILITY"] = round(sigma_v_raw, 5) if pd.notna(sigma_v_raw) else np.nan

    lnE = np.log(df["MARKET EQUITY VALUE"].replace(0, np.nan))
    se_global = lnE.diff().std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    if np.isnan(se_global) or se_global <= 0:
        se_global = SE_FALLBACK

    finalV = []
    per_row_sV = []
    for _, row in df.iterrows():
        E = row["MARKET EQUITY VALUE"]
        D = row["DPT"]
        r_log_row = row.get("LOG RISK FREE RATE", np.nan)
        if pd.isna(E) or pd.isna(D) or D <= 0 or pd.isna(r_log_row):
            finalV.append(np.nan)
            per_row_sV.append(np.nan)
            continue
        v_sol, sV_sol = kmv_solve(E, D, r_log_row, T, se_global, v0=E, sv0=se_global)
        finalV.append(v_sol)
        per_row_sV.append(sV_sol)

    df["FINAL MV OF ASSET"] = finalV
    df["FINAL_ASSET_VOL_PER_ROW"] = per_row_sV

    filled_final = df["FINAL MV OF ASSET"].ffill()
    filled_final = filled_final.fillna(df["INITIAL MV OF ASSET"])
    df["FINAL MV OF ASSET"] = filled_final

    series_final = df["FINAL MV OF ASSET"].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    final_asset_return = series_final.pct_change()
    if len(final_asset_return) > 0:
        final_asset_return.iloc[0] = np.nan
    df["FINAL ASSET RETURN"] = final_asset_return

    returns_nonan = df["FINAL ASSET RETURN"].dropna()
    stdev_sample = returns_nonan.std(ddof=1) if len(returns_nonan) > 0 else np.nan
    final_asset_vol_raw = np.sqrt(TRADING_DAYS_PER_YEAR) * stdev_sample if pd.notna(stdev_sample) else np.nan
    final_asset_vol_display = round(final_asset_vol_raw, 5) if pd.notna(final_asset_vol_raw) else np.nan
    df["FINAL MV OF ASSET VOLATILITY"] = final_asset_vol_display

    df["d1"] = df.apply(
        lambda r: compute_d1(r["INITIAL MV OF ASSET"], r["DPT"], sigma_v_raw, r["LOG RISK FREE RATE"], T),
        axis=1
    )
    df["d2"] = df.apply(
        lambda r: (r["d1"] - (sigma_v_raw * sqrt(T) if pd.notna(sigma_v_raw) else np.nan)),
        axis=1
    )

    df["IMPLIED MARKET EQUITY VALUE"] = df.apply(
        lambda r: (
            r["INITIAL MV OF ASSET"] * norm.cdf(r["d1"])
            - r["DPT"] * np.exp(-r["LOG RISK FREE RATE"] * T) * norm.cdf(r["d2"])
        ) if pd.notna(r["d1"]) and pd.notna(r["d2"]) and pd.notna(r["LOG RISK FREE RATE"]) else np.nan,
        axis=1
    )

    final_expected_return = df["FINAL ASSET RETURN"].dropna().mean() if df["FINAL ASSET RETURN"].dropna().size > 0 else np.nan
    df["FINAL EXPECTED RETURN"] = final_expected_return

    mv_asset_end = df["FINAL MV OF ASSET"].dropna().iloc[-1] if df["FINAL MV OF ASSET"].dropna().size > 0 else np.nan
    df["MV OF ASSET AT END"] = mv_asset_end

    dpt_end = df["DPT"].dropna().iloc[-1] if df["DPT"].dropna().size > 0 else np.nan
    df["DPT AT END"] = dpt_end

    coverage_ratio = (mv_asset_end / dpt_end) if pd.notna(mv_asset_end) and pd.notna(dpt_end) and dpt_end != 0 else np.nan
    df["COVERAGE RATIO"] = coverage_ratio

    df["ASSET DRIFT RATE"] = final_expected_return
    df["ASSET VOLATILITY"] = final_asset_vol_display
    df["TIME (YEAR)"] = T

    if pd.notna(coverage_ratio) and pd.notna(final_expected_return) and pd.notna(final_asset_vol_raw) and final_asset_vol_raw != 0:
        dd = (log(coverage_ratio) + (final_expected_return - 0.5 * final_asset_vol_raw * final_asset_vol_raw) * T) / (final_asset_vol_raw * sqrt(T))
    else:
        dd = np.nan
    df["DISTANCE TO DEFAULT"] = dd

    # Compute default probability (no clamping)
    df["DEFAULT PROBABILITY"] = df["DISTANCE TO DEFAULT"].apply(lambda x: norm.cdf(-x) if pd.notna(x) else np.nan)

    return df


# --------------------- Export ---------------------
def format_and_export(df: pd.DataFrame, output_dir: Path, excel_path: Path) -> Path:
    """
    Format columns to text-friendly representation and export to CSV.
    Keeps the same output column set as the original script.
    """
    out_cols = [
        "DATE", "MARKET EQUITY VALUE", "LONG TERM DEBT", "SHORT TERM DEBT",
        "RISK FREE RATE (%)", "INITIAL MV OF ASSET", "MV OF ASSET RETURN",
        "INITIAL MV OF ASSET VOLATILITY", "DPT", "LOG RISK FREE RATE",
        "TIME HORIZON", "d1", "d2", "IMPLIED MARKET EQUITY VALUE",
        "FINAL MV OF ASSET", "FINAL MV OF ASSET VOLATILITY", "FINAL ASSET RETURN",
        "FINAL EXPECTED RETURN", "MV OF ASSET AT END", "DPT AT END",
        "COVERAGE RATIO", "ASSET DRIFT RATE", "ASSET VOLATILITY",
        "TIME (YEAR)", "DISTANCE TO DEFAULT", "DEFAULT PROBABILITY"
    ]

    for col in out_cols:
        if col not in df.columns:
            df[col] = np.nan

    out = df[out_cols].copy()

    out_formatted = out.copy()
    for col in out_formatted.columns:
        if col == "DATE":
            out_formatted[col] = out_formatted[col].dt.strftime("%Y-%m-%d")
        elif col in ["INITIAL MV OF ASSET VOLATILITY", "FINAL MV OF ASSET VOLATILITY", "ASSET VOLATILITY"]:
            out_formatted[col] = out_formatted[col].apply(lambda x: f"{x:.5f}" if pd.notna(x) else "")
        elif col == "DEFAULT PROBABILITY":
            # export default prob in scientific uppercase E with 5 significant digits
            out_formatted[col] = out_formatted[col].apply(lambda x: f"{x:.5E}" if pd.notna(x) else "")
        elif pd.api.types.is_numeric_dtype(out_formatted[col].dtype):
            if col == "LOG RISK FREE RATE":
                out_formatted[col] = out_formatted[col].apply(lambda x: f"{x:.9f}" if pd.notna(x) else "")
            else:
                out_formatted[col] = out_formatted[col].apply(lambda x: f"{x:.10f}" if pd.notna(x) else "")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = excel_path.stem
    out_path = output_dir / f"{dataset_name}_output.csv"
    out_formatted.to_csv(out_path, index=False)

    return out_path


# --------------------- Main ---------------------
def main() -> None:
    logger.info("Loading data from Excel...")
    df = load_and_prepare_dataframe(EXCEL_PATH, SHEET_NAME, MAX_ROWS)
    logger.info("Computing metrics...")
    df = compute_metrics(df)
    logger.info("Exporting results...")
    out_path = format_and_export(df, OUTPUT_DIR, EXCEL_PATH)
    logger.info(f"Saved output to: {out_path}")


if __name__ == "__main__":
    main()
