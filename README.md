# KMV–Merton Credit Risk Modelling Using Python

## Project Overview

This project presents a Python-based implementation of the KMV/Merton structural credit risk model to estimate corporate asset values, asset volatility, Distance to Default (DD), and model-implied Default Probability (PD).

Developed as part of an academic research project examining Malaysian publicly listed companies, the implementation processes financial datasets covering 2014–2023 and demonstrates the application of quantitative financial modelling and numerical optimisation techniques.

## Technologies Used

- **Python** — Financial modelling and computational implementation
- **pandas** — Financial data processing and manipulation
- **NumPy** — Numerical and statistical calculations
- **SciPy** — Nonlinear equation solving and probability estimation
- **openpyxl** — Excel file processing

## Model Methodology

The project applies the structural credit risk framework introduced by the Merton model.

The computational workflow consists of:

1. Importing and preparing company financial data from Excel datasets.
2. Processing market equity values, debt obligations, and risk-free rates.
3. Estimating the default point from short-term and long-term debt.
4. Solving nonlinear equations to estimate corporate asset values and volatility.
5. Calculating asset returns and annualised volatility.
6. Estimating Distance to Default (DD).
7. Deriving model-implied Default Probability (PD).
8. Exporting calculated financial metrics into structured CSV files.

## Project Implementation

### Multi-Company Credit Risk Analysis — `fyp_full.py`

Processes annual financial datasets containing multiple companies, performing company-specific KMV/Merton calculations and exporting financial risk metrics.

### Single-Company Model Testing — `fyp_single.py`

Implements the modelling workflow for a single company, supporting initial testing and examination of model calculations.

## Dataset Information

The original research utilised annual financial datasets covering 2014–2023, obtained through LSEG Workplace.

The original datasets and generated research outputs are excluded from this public repository due to potential data licensing restrictions.

## Technical Skills Demonstrated

- Python programming and computational financial modelling
- Financial data preparation and transformation
- Numerical optimisation using SciPy
- Statistical calculations using NumPy
- Multi-company data processing with pandas
- Automated financial risk metric calculations
- CSV-based results generation

## Model Limitations

The implementation uses assumptions concerning debt structure, asset volatility, and default conditions. Calculated default probabilities are model-implied estimates rather than independently calibrated real-world default frequencies.

The implementation requires further numerical testing and validation before use in financial decision-making.

## Project Purpose

This repository demonstrates the application of Python programming, numerical methods, and quantitative financial modelling to corporate credit risk analysis.

The project is presented for academic and technical portfolio purposes.