# ML-Trading-Strategy-BTC

# Quantitative ML Trading Strategy for BTC/USDT

A comprehensive research script for developing and validating a machine learning-based trading strategy for the BTC/USDT pair on the 15-minute timeframe.

## Project Overview

This project implements an end-to-end pipeline for creating a dual-model (Long/Short) trading system using LightGBM. The focus is not just on building a predictive model, but on ensuring its robustness through rigorous validation techniques, including hyperparameter optimization with cross-validation and walk-forward analysis.

## Key Features & Methodologies

*   **Advanced Feature Engineering:** Utilizes multi-timeframe analysis (15m, 1H, 4H) and incorporates advanced indicators like Choppiness Index and Volatility Ratio to capture diverse market regimes.
*   **Robust Hyperparameter Tuning:** Employs **Optuna** with **Stratified K-Fold Cross-Validation** to find the most optimal and generalizable model parameters, systematically reducing the risk of overfitting.
*   **High-Fidelity Backtesting Engine:** A custom event-driven backtester that simulates trades with high accuracy, accounting for:
    *   Trading Fees
    *   Dynamic Position Sizing based on portfolio risk
    *   Multiple exit conditions (SL/TP, Time Exit, AI-based counter-signal)
*   **Walk-Forward Validation:** Implements a rigorous walk-forward analysis to assess the strategy's performance stability across different time periods, simulating a real-world deployment and retraining cycle. This is a critical step to evaluate robustness against market concept drift.
*   **Performance Optimization:** Leverages **Numba** (`@njit`) to accelerate critical computational loops (e.g., label generation).

## How to Run

1.  **Clone the repository:**

2.  **Install dependencies:**
    pip install -r requirements.txt

3.  **Run the script:**
    python PhamCanh_Trading_System_Research.py
 
    The script will automatically download the necessary data from Binance if the local `.parquet` file is not found. It will then proceed with feature engineering, model training, validation, and backtesting, finally displaying the results and charts.
📊 FEATURE IMPORTANCE - LONG MODEL (V3 - OPTIMIZED CV)
         Feature  Importance (%)
          rsi_4h            9.96
          adx_1h            9.91
          rsi_1h            8.94
volatility_ratio            7.63
   dist_ema50_1h            6.72
         atr_pct            6.57
             adx            5.92
     dist_ema200            5.62
            chop            5.57
        bb_width            5.49
   bbw_expansion            5.33
             rsi            4.67

📊 FEATURE IMPORTANCE - SHORT MODEL (V3 - OPTIMIZED CV)
         Feature  Importance (%)
          rsi_4h           11.63
          adx_1h            9.27
          rsi_1h            8.97
         atr_pct            8.52
volatility_ratio            7.29
             adx            6.10
   dist_ema50_1h            5.98
        bb_width            5.70
   bbw_expansion            5.30
            chop            4.94
     dist_ema200            4.69
             rsi            4.60
##  
============================================================
 R RUNNING WALK-FORWARD VALIDATION TEST
============================================================

--- FOLD 1/5 ---
   Train period: 2024-11-13 08:00:00 -> 2025-01-14 06:00:00 (5098 samples)
   Test period:  2025-01-14 06:15:00 -> 2025-03-18 15:00:00 (5098 samples)
   ⏳ Re-training models for this fold...
   ✅ Fold 1 Results -> Long AUC: 0.7689 | Short AUC: 0.7533

--- FOLD 2/5 ---
   Train period: 2024-11-13 08:00:00 -> 2025-03-18 15:00:00 (10196 samples)
   Test period:  2025-03-18 15:15:00 -> 2025-05-23 07:45:00 (5098 samples)
   ⏳ Re-training models for this fold...
   ✅ Fold 2 Results -> Long AUC: 0.7583 | Short AUC: 0.7503

--- FOLD 3/5 ---
   Train period: 2024-11-13 08:00:00 -> 2025-05-23 07:45:00 (15294 samples)
   Test period:  2025-05-23 09:45:00 -> 2025-08-11 20:30:00 (5098 samples)
   ⏳ Re-training models for this fold...
   ✅ Fold 3 Results -> Long AUC: 0.8208 | Short AUC: 0.7841

--- FOLD 4/5 ---
   Train period: 2024-11-13 08:00:00 -> 2025-08-11 20:30:00 (20392 samples)
   Test period:  2025-08-11 20:45:00 -> 2025-10-31 07:30:00 (5098 samples)
   ⏳ Re-training models for this fold...
   ✅ Fold 4 Results -> Long AUC: 0.8104 | Short AUC: 0.7315

--- FOLD 5/5 ---
   Train period: 2024-11-13 08:00:00 -> 2025-10-31 07:30:00 (25490 samples)
   Test period:  2025-10-31 07:45:00 -> 2026-01-12 04:00:00 (5098 samples)
   ⏳ Re-training models for this fold...
   ✅ Fold 5 Results -> Long AUC: 0.8016 | Short AUC: 0.8010

============================================================
📊 WALK-FORWARD VALIDATION SUMMARY
============================================================
 Fold Test_Start   Test_End  AUC_Long  AUC_Short
    1 2025-01-14 2025-03-18    0.7689     0.7533
    2 2025-03-18 2025-05-23    0.7583     0.7503
    3 2025-05-23 2025-08-11    0.8208     0.7841
    4 2025-08-11 2025-10-31    0.8104     0.7315
    5 2025-10-31 2026-01-12    0.8016     0.8010

--- Overall Performance ---
📈 AVERAGE Long AUC:  0.7920 (Std Dev: 0.0271)
📉 AVERAGE Short AUC: 0.7640 (Std Dev: 0.0280)

--- Interpretation ---
   ✅ GOOD: The model's performance is relatively stable across periods (low Std Dev).
   ✅ GOOD: The model's average performance is at an acceptable/good level.
(.venv) bot@bot-All-Series:~/freqtrade$ 


