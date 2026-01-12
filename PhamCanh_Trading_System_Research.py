# --- START OF FILE Btclongshortv4_1live06012025.py ---

# --- START OF FILE   (MODIFIED VERSION - V2) ---

# --- START OF FILE uworkcorinix_V3_Advanced.py ---

import pandas as pd
import numpy as np
import talib as ta
import lightgbm as lgb
import os
import requests
import matplotlib.pyplot as plt
import warnings
from numba import njit
from sklearn.metrics import roc_auc_score
import joblib

# Disable warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# =============================================================================
# STRATEGY CONFIGURATION (CONFIG) - "HIGH-QUALITY V3" VERSION
# =============================================================================
SYMBOL = 'BTCUSDT'
TIMEFRAME = '15m'
CANDLE_LIMIT = 50000 # Increase data for better model learning
CAPITAL = 1000
LEVERAGE = 10
RISK_PER_TRADE = 0.05
STOP_LOSS_ATR = 2.0
TAKE_PROFIT_ATR = 4.0
MIN_PROBABILITY = 0.60

# =============================================================================
# BLOCK 1: DATA LOADING (UNCHANGED)
# =============================================================================
print(f"--- 1. LOADING DATA: {SYMBOL} ({TIMEFRAME}) ---")
data_filename = f"{SYMBOL}_{TIMEFRAME}_v5_opt.parquet"

def fetch_binance_data(symbol, interval, limit, total):
    # ... (data fetching code remains the same)
    print(f"⏳ Fetching {total} candles from Binance...")
    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = None
    loops = int(total / limit) + 1
    
    for _ in range(loops):
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        if end_time: params['endTime'] = end_time
        try:
            r = requests.get(base_url, params=params)
            data = r.json()
            if not data: break
            all_data = data + all_data
            end_time = data[0][0] - 1
        except Exception as e:
            print(f"Error: {e}"); break
            
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df.set_index('timestamp', inplace=True)
    return df[~df.index.duplicated(keep='last')].sort_index()

if not os.path.exists(data_filename):
    df_15m = fetch_binance_data(SYMBOL, TIMEFRAME, 1000, CANDLE_LIMIT)
    df_15m.to_parquet(data_filename)
else:
    df_15m = pd.read_parquet(data_filename)
    print(f"✅ Loaded local file: {len(df_15m)} candles.")

agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
df_1h = df_15m.resample('1h').agg(agg).dropna()
df_4h = df_15m.resample('4h').agg(agg).dropna()

# =============================================================================
# BLOCK 2: FEATURE ENGINEERING - V3 "ADVANCED CONTEXT" VERSION
# =============================================================================
print("--- 2. CREATING ADVANCED CONTEXT FEATURES (V3) ---")

def add_base_features(df_input, df_1h, df_4h):
    # ... (This function remains the same as in V2)
    df = df_input.copy()
    
    # 1. Higher Timeframe Context (4H & 1H) - Kept as a foundation
    df_4h['ema200_4h'] = ta.EMA(df_4h['close'], 200)
    df_4h['trend_dir_4h'] = np.where(df_4h['close'] > df_4h['ema200_4h'], 1, -1)
    df_4h['rsi_4h'] = ta.RSI(df_4h['close'], 14)
    df = pd.merge_asof(df, df_4h[['trend_dir_4h', 'rsi_4h']].sort_index(), 
                       left_index=True, right_index=True, direction='backward')

    df_1h['ema50_1h'] = ta.EMA(df_1h['close'], 50)
    df_1h['rsi_1h'] = ta.RSI(df_1h['close'], 14)
    df_1h['adx_1h'] = ta.ADX(df_1h['high'], df_1h['low'], df_1h['close'], 14)
    df = pd.merge_asof(df, df_1h[['ema50_1h', 'rsi_1h', 'adx_1h']].sort_index(),
                       left_index=True, right_index=True, direction='backward')
    df['dist_ema50_1h'] = (df['close'] - df['ema50_1h']) / df['ema50_1h']
    
    # 2. Current Timeframe Features (15m) - Retaining core features
    df['atr'] = ta.ATR(df['high'], df['low'], df['close'], 14)
    df['atr_pct'] = df['atr'] / df['close']
    df['adx'] = ta.ADX(df['high'], df['low'], df['close'], 14)
    df['rsi'] = ta.RSI(df['close'], 14)
    df['ema200'] = ta.EMA(df['close'], 200)
    df['dist_ema200'] = (df['close'] - df['ema200']) / df['ema200']
    upper, mid, lower = ta.BBANDS(df['close'], 20, 2, 2)
    df['bb_width'] = (upper - lower) / mid
    
    return df

def add_momentum_features(df_input):
    # ... (This function remains the same as in V2)
    df = df_input.copy()
    # 1. Measure momentum "acceleration"
    df['rsi_roc'] = ta.ROC(df['rsi'], timeperiod=3)
    # 2. Detect unusual Volume spikes
    vol_ma = df['volume'].rolling(window=30).mean()
    vol_std = df['volume'].rolling(window=30).std()
    df['volume_zscore'] = (df['volume'] - vol_ma) / vol_std
    # 3. Detect volatility breakouts
    df['bbw_ma'] = df['bb_width'].rolling(window=20).mean()
    df['bbw_expansion'] = (df['bb_width'] - df['bbw_ma']) / df['bbw_ma']
    # 4. Measure short-term trend slope
    df['ema_slope'] = ta.LINEARREG_SLOPE(ta.EMA(df['close'], 20), timeperiod=5)
    # 5. Interaction feature: combination of major trend and minor acceleration
    df['ema_slope_x_trend'] = df['ema_slope'] * df['trend_dir_4h']
    
    return df
    
# ### <<< V3 IMPROVEMENT: ADD ADVANCED CONTEXT FEATURES >>>
def add_advanced_context_features(df_input):
    """Adds features related to market nature and volatility."""
    df = df_input.copy()
    print("   + Adding Advanced Context features (CHOP, Volatility Ratio)...")
    
    # 1. Choppiness Index (CHOP) - Measures if the market is trending or sideways
    # High CHOP (~>61.8) -> Sideways. Low CHOP (~<38.2) -> Trending.
    atr1 = ta.ATR(df['high'], df['low'], df['close'], 1)
    highest_high = df['high'].rolling(window=14).max()
    lowest_low = df['low'].rolling(window=14).min()
    chop_numerator = atr1.rolling(window=14).sum()
    chop_denominator = highest_high - lowest_low
    df['chop'] = 100 * np.log10(chop_numerator / chop_denominator) / np.log10(14)
    
    # 2. Volatility Ratio - Compares short-term vs. long-term volatility
    atr_long = ta.ATR(df['high'], df['low'], df['close'], 100)
    # atr_short is already calculated as 'atr' (period 14)
    df['volatility_ratio'] = df['atr'] / atr_long
    
    # IMPROVED SIDEWAYS FILTER
    bbw_squeeze_threshold = df['bb_width'].rolling(100).quantile(0.15)
    # Add CHOP > 60 condition to confirm sideways market
    df['is_sideways'] = (df['adx'] < 20) & (df['bb_width'] < bbw_squeeze_threshold) & (df['chop'] > 60)
    
    return df.dropna()

df_base = add_base_features(df_15m, df_1h, df_4h)
df_momentum = add_momentum_features(df_base)
df_features = add_advanced_context_features(df_momentum)

## =============================================================================
# BLOCK 3: LABELING, OPTIMIZATION & TRAINING (V3 + OPTUNA)
# =============================================================================
print("--- 3. LABELING, OPTIMIZING & TRAINING ---")

@njit
def get_label_fast(prices_high, prices_low, entry_prices, tps, sls, window):
    # ... (This function remains unchanged)
    n = len(entry_prices)
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n):
        entry_price = entry_prices[i]
        tp = tps[i]
        sl = sls[i]
        for j in range(1, window + 1):
            if i + j >= n: break 
            high, low = prices_high[i+j], prices_low[i+j]
            if tp > sl: # LONG
                if low <= sl: labels[i] = 0; break
                if high >= tp: labels[i] = 1; break
            else: # SHORT
                if high >= sl: labels[i] = 0; break
                if low <= tp: labels[i] = 1; break
    return labels

def create_labels_v3(df):
    # ... (This function remains unchanged)
    df = df.copy()
    window_size = 8
    atr_val = df['atr'].values
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    tp_long = close_prices + atr_val * TAKE_PROFIT_ATR
    sl_long = close_prices - atr_val * STOP_LOSS_ATR
    tp_short = close_prices - atr_val * TAKE_PROFIT_ATR
    sl_short = close_prices + atr_val * STOP_LOSS_ATR
    print(f"   ⏳ Generating STRICT labels (TP/SL within {window_size} bars)...")
    df['y_long'] = get_label_fast(high_prices, low_prices, close_prices, tp_long, sl_long, window_size)
    df['y_short'] = get_label_fast(high_prices, low_prices, close_prices, tp_short, sl_short, window_size)
    print("   ✅ Labeling complete.")
    return df

df_labeled = create_labels_v3(df_features)

volatility_threshold = df_labeled['bb_width'].quantile(0.25)
df_filtered = df_labeled[df_labeled['bb_width'] > volatility_threshold].copy()
print(f"🔬 Filtering data: Keeping {len(df_filtered)} / {len(df_labeled)} samples ({len(df_filtered)/len(df_labeled):.1%}) with high volatility.")

features_list_advanced = [
    'trend_dir_4h', 'rsi_4h', 'rsi_1h', 'adx_1h', 'dist_ema50_1h', 'dist_ema200',
    'adx', 'rsi', 'bb_width', 'atr_pct',
    'rsi_roc', 'volume_zscore', 'bbw_expansion', 'ema_slope', 'ema_slope_x_trend',
    'chop', 'volatility_ratio'
]
print(f"🕵️  Using {len(features_list_advanced)} advanced features.")

X = df_filtered[features_list_advanced]
y_long = df_filtered['y_long']
y_short = df_filtered['y_short']

train_split = int(len(X) * 0.70)
val_split = int(len(X) * 0.85)
X_train, X_val, X_test = X.iloc[:train_split], X.iloc[train_split:val_split], X.iloc[val_split:]
y_long_train, y_long_val, y_long_test = y_long.iloc[:train_split], y_long.iloc[train_split:val_split], y_long.iloc[val_split:]
y_short_train, y_short_val, y_short_test = y_short.iloc[:train_split], y_short.iloc[train_split:val_split], y_short.iloc[val_split:]

def calculate_scale_pos_weight(y_series):
    counts = y_series.value_counts()
    return counts.get(0, 0) / counts.get(1, 1) if counts.get(1, 0) > 0 else 1.0

spw_long = calculate_scale_pos_weight(y_long_train)
spw_short = calculate_scale_pos_weight(y_short_train)
print(f"\n⚖️  Automatically adjusting data balance:")
print(f"   - Long model scale_pos_weight: {spw_long:.2f}")
print(f"   - Short model scale_pos_weight: {spw_short:.2f}")

# =============================================================================
# BLOCK 3: LABELING, OPTIMIZING (CV) & TRAINING
# =============================================================================
print("--- 3. LABELING, OPTIMIZING (CV) & TRAINING ---")

# --- PART 1: LABELING ---
@njit
def get_label_fast(prices_high, prices_low, entry_prices, tps, sls, window):
    n = len(entry_prices)
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n):
        entry_price = entry_prices[i]
        tp = tps[i]
        sl = sls[i]
        for j in range(1, window + 1):
            if i + j >= n: break 
            high, low = prices_high[i+j], prices_low[i+j]
            if tp > sl: # LONG
                if low <= sl: labels[i] = 0; break
                if high >= tp: labels[i] = 1; break
            else: # SHORT
                if high >= sl: labels[i] = 0; break
                if low <= tp: labels[i] = 1; break
    return labels

def create_labels_v3(df):
    df = df.copy()
    window_size = 8
    atr_val = df['atr'].values
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    tp_long = close_prices + atr_val * TAKE_PROFIT_ATR
    sl_long = close_prices - atr_val * STOP_LOSS_ATR
    tp_short = close_prices - atr_val * TAKE_PROFIT_ATR
    sl_short = close_prices + atr_val * STOP_LOSS_ATR
    print(f"   ⏳ Generating STRICT labels (TP/SL within {window_size} bars)...")
    df['y_long'] = get_label_fast(high_prices, low_prices, close_prices, tp_long, sl_long, window_size)
    df['y_short'] = get_label_fast(high_prices, low_prices, close_prices, tp_short, sl_short, window_size)
    print("   ✅ Labeling complete.")
    return df

df_labeled = create_labels_v3(df_features)

# --- PART 2: FILTERING AND SPLITTING DATA ---
volatility_threshold = df_labeled['bb_width'].quantile(0.25)
df_filtered = df_labeled[df_labeled['bb_width'] > volatility_threshold].copy()
print(f"🔬 Filtering data: Keeping {len(df_filtered)} / {len(df_labeled)} samples ({len(df_filtered)/len(df_labeled):.1%}) with high volatility.")

features_list_advanced = [
    'trend_dir_4h', 'rsi_4h', 'rsi_1h', 'adx_1h', 'dist_ema50_1h', 'dist_ema200',
    'adx', 'rsi', 'bb_width', 'atr_pct',
    'rsi_roc', 'volume_zscore', 'bbw_expansion', 'ema_slope', 'ema_slope_x_trend',
    'chop', 'volatility_ratio'
]
print(f"🕵️  Using {len(features_list_advanced)} advanced features.")

X = df_filtered[features_list_advanced]
y_long = df_filtered['y_long']
y_short = df_filtered['y_short']

train_split = int(len(X) * 0.70)
val_split = int(len(X) * 0.85)
X_train, X_val, X_test = X.iloc[:train_split], X.iloc[train_split:val_split], X.iloc[val_split:]
y_long_train, y_long_val, y_long_test = y_long.iloc[:train_split], y_long.iloc[train_split:val_split], y_long.iloc[val_split:]
y_short_train, y_short_val, y_short_test = y_short.iloc[:train_split], y_short.iloc[train_split:val_split], y_short.iloc[val_split:]

def calculate_scale_pos_weight(y_series):
    counts = y_series.value_counts()
    return counts.get(0, 0) / counts.get(1, 1) if counts.get(1, 0) > 0 else 1.0

spw_long = calculate_scale_pos_weight(y_long_train)
spw_short = calculate_scale_pos_weight(y_short_train)
print(f"\n⚖️  Automatically adjusting data balance:")
print(f"   - Long model scale_pos_weight: {spw_long:.2f}")
print(f"   - Short model scale_pos_weight: {spw_short:.2f}")


# --- PART 3: OPTIMIZATION WITH OPTUNA AND CROSS-VALIDATION ---

# --- START: INTEGRATING OPTUNA WITH CROSS-VALIDATION (CV) TO FIND BEST HYPERPARAMETERS ---
import optuna
from sklearn.model_selection import StratifiedKFold
import numpy as np
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Define a COMMON Objective Function using Cross-Validation
def objective_cv(trial, X_data, y_data, scale_pos_weight_val):
    """
    Common objective function for Optuna, using Stratified K-Fold Cross-Validation.
    This function will be called for both the Long and Short models.
    """
    # Parameter search space - adjusted for stronger overfitting prevention
    params = {
        'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'n_jobs': -1, 'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2500, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 7), 
        'num_leaves': trial.suggest_int('num_leaves', 8, 40),
        'min_child_samples': trial.suggest_int('min_child_samples', 300, 800),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
        'subsample': trial.suggest_float('subsample', 0.4, 0.8),
        'scale_pos_weight': scale_pos_weight_val
    }

    N_SPLITS = 5
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=trial.number)
    auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_data, y_data)):
        X_tr, X_va = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_tr, y_va = y_data.iloc[train_idx], y_data.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, 
                  eval_set=[(X_va, y_va)], 
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        preds = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, preds)
        auc_scores.append(auc)

    return np.mean(auc_scores)

# --- Run optimization ---
N_TRIALS_CV = 50 
X_opt = X_train
y_long_opt = y_long_train
y_short_opt = y_short_train

# 2. Run optimization for the LONG model
print(f"\n⚙️  Starting CV optimization process for LONG model ({N_TRIALS_CV} trials x 5 folds)...")
study_long = optuna.create_study(direction='maximize')
study_long.optimize(lambda trial: objective_cv(trial, X_opt, y_long_opt, spw_long), n_trials=N_TRIALS_CV)
best_params_long = study_long.best_params
print(f"   ✅ Optimization finished! Best average AUC: {study_long.best_value:.4f}")

# 3. Run optimization for the SHORT model
print(f"\n⚙️  Starting CV optimization process for SHORT model ({N_TRIALS_CV} trials x 5 folds)...")
study_short = optuna.create_study(direction='maximize')
study_short.optimize(lambda trial: objective_cv(trial, X_opt, y_short_opt, spw_short), n_trials=N_TRIALS_CV)
best_params_short = study_short.best_params
print(f"   ✅ Optimization finished! Best average AUC: {study_short.best_value:.4f}")

# --- END: OPTUNA WITH CROSS-VALIDATION INTEGRATION ---


# --- PART 4: TRAINING FINAL MODEL AND SAVING FILE ---

# Combine train and validation data to train a more robust final model
X_train_full = pd.concat([X_train, X_val])
y_long_train_full = pd.concat([y_long_train, y_long_val])
y_short_train_full = pd.concat([y_short_train, y_short_val])

# Update base parameters and scale_pos_weight into the best parameter set from Optuna
final_params_long = {'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'n_jobs': -1, 'random_state': 42, 'scale_pos_weight': spw_long}
final_params_long.update(best_params_long)

final_params_short = {'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'n_jobs': -1, 'random_state': 42, 'scale_pos_weight': spw_short}
final_params_short.update(best_params_short)

print("\n   ⚡ Training FINAL Long Model with OPTIMIZED PARAMS (CV)...")
print("   ", final_params_long)
model_long = lgb.LGBMClassifier(**final_params_long)
# Train on the entire dataset (train+val), no early stopping needed
model_long.fit(X_train_full, y_long_train_full) 

print("\n   ⚡ Training FINAL Short Model with OPTIMIZED PARAMS (CV)...")
print("   ", final_params_short)
model_short = lgb.LGBMClassifier(**final_params_short)
model_short.fit(X_train_full, y_short_train_full)

def show_importance(model, name, features_list):
    imp_df = pd.DataFrame({'Feature': features_list, 'Importance': model.feature_importances_})
    imp_df = imp_df.sort_values(by='Importance', ascending=False)
    imp_df['Importance (%)'] = (imp_df['Importance'] / imp_df['Importance'].sum()) * 100
    print(f"\n📊 FEATURE IMPORTANCE - {name} MODEL (V3 - OPTIMIZED CV)")
    print(imp_df[['Feature', 'Importance (%)']].head(12).to_string(index=False, float_format="%.2f"))

show_importance(model_long, "LONG", features_list_advanced)
show_importance(model_short, "SHORT", features_list_advanced)

print("\n--- SAVING DUAL-MODEL V3 (OPTIMIZED CV) FOR FREQTRADE STRATEGY ---")
# Rename the model file to clearly distinguish the CV-optimized version
model_filename = 'lgb_dual_model_v3_optimized_cv.pkl' 
model_dir = "user_data/models"
os.makedirs(model_dir, exist_ok=True)
full_path = os.path.join(model_dir, model_filename)

data_to_save = {
    'model_long': model_long,
    'model_short': model_short,
    'features': features_list_advanced,
    'trained_on_pair': SYMBOL,
    'trained_on_timeframe': TIMEFRAME
}
joblib.dump(data_to_save, full_path)
print(f"✅ Successfully packaged and saved OPTIMIZED CV model V3 to: {full_path}")
print("   -> Copy this file to user_data/models/ and update MODEL_FILENAME in your strategy.")


# --- PART 5: FINAL OVERFITTING ANALYSIS ---
def analyze_overfitting_optimized(model, X_tr_full, y_tr_full, X_te, y_te, label_name):
    pred_train = model.predict_proba(X_tr_full)[:, 1]
    pred_test = model.predict_proba(X_te)[:, 1]

    auc_train = roc_auc_score(y_tr_full, pred_train)
    auc_test = roc_auc_score(y_te, pred_test)

    print(f"\n📊 CV-OPTIMIZED MODEL EVALUATION: {label_name}")
    print(f"   🔹 Train+Val AUC: {auc_train:.4f}")
    print(f"   🔹 Test  AUC: {auc_test:.4f} (MOST IMPORTANT)")
    gap = auc_train - auc_test
    print(f"   ⚠️ Gap (Train+Val - Test): {gap:.4f}")
    if gap > 0.1: 
        print(f"   WARNING: Signs of overfitting still present.")
    elif gap < 0.05:
        print(f"   ✅ EXCELLENT: Model shows very good generalization capability.")
    else:
        print(f"   ✅ GOOD: Model generalizes well, gap is acceptable.")

analyze_overfitting_optimized(model_long, X_train_full, y_long_train_full, X_test, y_long_test, "LONG")
analyze_overfitting_optimized(model_short, X_train_full, y_short_train_full, X_test, y_short_test, "SHORT")

# =============================================================================
# BLOCK 4: HIGH-FIDELITY BACKTEST (V3.1)
# =============================================================================
print("\n--- 4. RUNNING HIGH-FIDELITY SIMULATION (V3.1) ---")

# --- BACKTEST CONFIGURATION ---
TRADING_FEE = 0.0005 # 0.05% fee for each order execution (maker/taker)

# Prepare data for backtesting
# Important: Always test on the full, unfiltered dataset to simulate reality
sim_data = df_labeled[df_labeled.index >= X_test.index[0]].copy()

print("   ⏳ Predicting signals for backtest period...")
X_sim = sim_data[features_list_advanced]
sim_data['prob_long'] = model_long.predict_proba(X_sim)[:, 1]
sim_data['prob_short'] = model_short.predict_proba(X_sim)[:, 1]

thresh_long = max(np.percentile(sim_data['prob_long'], 97), MIN_PROBABILITY)
thresh_short = max(np.percentile(sim_data['prob_short'], 97), MIN_PROBABILITY)
print(f"🔥 Activation Thresholds (Top 3%): Long > {thresh_long:.3f} | Short > {thresh_short:.3f}")

equity = [CAPITAL]
active_trade = None # Instead of a list, allow only one active trade at a time
trade_history = [] 

class CLR:
    GREEN = '\033[92m'; RED = '\033[91m'; YELLOW = '\033[93m'; BLUE = '\033[94m'; END = '\033[0m'

for idx, row in sim_data.iterrows():
    current_eq = equity[-1]
    
    # 1. Manage active trade
    if active_trade:
        t = active_trade
        t['bars'] += 1
        pnl = 0; closed = False; reason = ""; exit_price = row['close']
        
        # SL/TP logic (check if the current candle hits SL/TP)
        if t['side'] == 'LONG':
            if row['low'] <= t['sl']: 
                exit_price = t['sl']; closed=True; reason="🛑 SL"
            elif row['high'] >= t['tp']: 
                exit_price = t['tp']; closed=True; reason="✅ TP"
        else: # SHORT
            if row['high'] >= t['sl']: 
                exit_price = t['sl']; closed=True; reason="🛑 SL"
            elif row['low'] <= t['tp']: 
                exit_price = t['tp']; closed=True; reason="✅ TP"

        # Other exit logic
        if not closed:
            is_ai_exit = False
            if (t['side'] == 'LONG' and row['prob_short'] > 0.65): is_ai_exit = True; reason = "🤖 AI Short"
            elif (t['side'] == 'SHORT' and row['prob_long'] > 0.65): is_ai_exit = True; reason = "🤖 AI Long"
            
            if is_ai_exit:
                closed = True
            elif t['bars'] > 32:
                closed = True; reason="⏱ Time"

        if closed:
            # Calculate accurate PnL
            pnl_multiplier = 1 if t['side'] == 'LONG' else -1
            gross_pnl = (exit_price - t['entry']) * t['size_asset'] * pnl_multiplier
            
            # Calculate trading fees
            entry_cost = t['size_usd'] * TRADING_FEE
            exit_cost = (t['size_asset'] * exit_price) * TRADING_FEE
            total_fees = entry_cost + exit_cost
            
            net_pnl = gross_pnl - total_fees
            
            current_eq += net_pnl
            
            clr = CLR.GREEN if net_pnl > 0 else CLR.RED
            print(f"   {clr}{t['side']} {reason}{CLR.END} | PnL: ${net_pnl:,.2f} (Fees: ${total_fees:,.2f}) | Equity: ${current_eq:,.2f}")
            trade_history.append({
                'entry_time': t['entry_time'], 'exit_time': idx, 'side': t['side'], 
                'pnl': net_pnl, 'reason': reason, 'entry_prob': t['entry_prob'],
                'bars_held': t['bars']
            })
            active_trade = None
    
    equity.append(current_eq)
    if current_eq <= 0: print("☠️  ACCOUNT BLOWN!"); break

    # 2. New entry logic (only enter if there's no active trade)
    if not active_trade and not row['is_sideways']:
        is_long_signal = row['prob_long'] > thresh_long and row['prob_short'] < 0.5
        is_short_signal = row['prob_short'] > thresh_short and row['prob_long'] < 0.5
        
        trade_side = None
        if is_long_signal: trade_side = 'LONG'
        if is_short_signal: trade_side = 'SHORT'
        
        if trade_side:
            entry_price = row['close']
            risk_amt_per_trade = current_eq * RISK_PER_TRADE
            
            if trade_side == 'LONG':
                sl_price = entry_price - (row['atr'] * STOP_LOSS_ATR)
                tp_price = entry_price + (row['atr'] * TAKE_PROFIT_ATR)
                prob = row['prob_long']
            else: #SHORT
                sl_price = entry_price + (row['atr'] * STOP_LOSS_ATR)
                tp_price = entry_price - (row['atr'] * TAKE_PROFIT_ATR)
                prob = row['prob_short']

            sl_distance_usd = abs(entry_price - sl_price)
            if sl_distance_usd == 0: continue
            
            # Calculate position size
            position_size_asset = risk_amt_per_trade / sl_distance_usd
            position_size_usd = position_size_asset * entry_price
            
            # Apply leverage
            capital_required = position_size_usd / LEVERAGE
            
            # Check for sufficient capital to open the trade
            if capital_required > current_eq:
                # print(f"[{idx}] {CLR.YELLOW}Skip: Not enough capital.{CLR.END}")
                continue

            active_trade = {
                'side': trade_side, 'entry': entry_price, 'sl': sl_price, 'tp': tp_price,
                'bars': 0, 'entry_time': idx, 'entry_prob': prob, 
                'size_asset': position_size_asset, 'size_usd': position_size_usd
            }
            print(f"[{idx}] {CLR.BLUE}⚡ ENTRY {trade_side}{CLR.END} | Price: {entry_price:,.2f} | Size: ${position_size_usd:,.2f} | Equity: ${current_eq:,.2f}")

# Close the last trade if it's still open at the end of the backtest
if active_trade:
    # ... (Logic to close the final trade can be added here if needed)
    pass
    
trade_history_df = pd.DataFrame(trade_history)

# =============================================================================
# BLOCK 5: RESULTS AND IN-DEPTH ANALYSIS (V3.1)
# =============================================================================
print("\n" + "="*50)
print("📊 SUMMARY REPORT (HIGH-FIDELITY MODEL)")
print("="*50)

if not trade_history_df.empty:
    equity_series = pd.Series(equity, index=pd.to_datetime(sim_data.index.tolist() + [sim_data.index[-1] + pd.Timedelta(minutes=15)]))
    
    # Basic metrics
    final_equity = equity_series.iloc[-1]
    total_return = (final_equity - CAPITAL) / CAPITAL * 100
    total_trades = len(trade_history_df)
    
    # Winrate & Payoff
    wins = trade_history_df[trade_history_df['pnl'] > 0]
    losses = trade_history_df[trade_history_df['pnl'] <= 0]
    winrate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

    print(f"🕒 Backtest Period: {sim_data.index[0]} -> {sim_data.index[-1]}")
    print(f"💰 Starting Capital: ${CAPITAL:,.2f} | Final Equity: ${final_equity:,.2f}")
    print(f"🚀 Total Return: {total_return:.2f}%")
    print("-" * 30)
    print(f"📈 Total Trades: {total_trades}")
    print(f"🎯 Winrate: {winrate:.2f}% ({len(wins)}/{total_trades})")
    print(f"🏆 Payoff Ratio: {payoff_ratio:.2f}")
    
    # Drawdown Analysis
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = -drawdown.min() * 100
    
    print(f"💣 Max Drawdown: {max_drawdown:.2f}%")

    # Sharpe Ratio (Assuming risk-free rate = 0)
    returns = equity_series.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365 * (24*4)) # Annualized
    print(f"⭐ Sharpe Ratio (annualized): {sharpe_ratio:.2f}")

    print("\n--- Analysis by Exit Reason ---")
    reason_analysis = trade_history_df.groupby('reason')['pnl'].agg(['count', 'sum', 'mean']).sort_values(by='sum', ascending=False)
    print(reason_analysis)
    print("="*50)

    # Plotting charts
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Equity Chart
    ax1.plot(equity_series.index, equity_series.values, label='Equity Curve', color='cyan')
    ax1.set_title(f"Equity Curve - Final ROI: {total_return:.2f}% | Max DD: {max_drawdown:.2f}% | Winrate: {winrate:.2f}%", fontsize=16)
    ax1.set_ylabel("Equity ($)", fontsize=12)
    ax1.grid(True)
    ax1.legend()

    # Drawdown Chart
    ax2.fill_between(drawdown.index, drawdown.values * 100, color='red', alpha=0.3)
    ax2.set_title("Drawdown Curve (%)", fontsize=14)
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.grid(True)

    plt.xlabel("Date", fontsize=12)
    plt.tight_layout()
    plt.show()

else:
    print("No trades were executed during the backtest period.")


# =============================================================================
# DEDICATED BACKTEST BLOCK: MONTHLY SIMULATION (V3.1)
# =============================================================================
print("\n" + "="*60)
print("📅 RUNNING BACKTEST FOR THE LAST YEAR (CAPITAL RESET TO $200 MONTHLY)")
print("="*60)

days_back = 365
start_date = df_labeled.index.max() - pd.Timedelta(days=days_back)
df_bt = df_labeled[df_labeled.index >= start_date].copy()

# --- MONTHLY BACKTEST CONFIGURATION ---
MONTHLY_CAPITAL = 200.0
TRADING_FEE = 0.0005 # 0.05% Fee

print(f"⏳ Running AI predictions for {len(df_bt)} candles ({days_back} days)...")
df_bt['prob_long'] = model_long.predict_proba(df_bt[features_list_advanced])[:, 1]
df_bt['prob_short'] = model_short.predict_proba(df_bt[features_list_advanced])[:, 1]

bt_thresh_long = max(np.percentile(df_bt['prob_long'], 97), MIN_PROBABILITY)
bt_thresh_short = max(np.percentile(df_bt['prob_short'], 97), MIN_PROBABILITY)
print(f"🔹 Long Threshold: {bt_thresh_long:.4f} | Short Threshold: {bt_thresh_short:.4f}")

df_bt['month_group'] = df_bt.index.to_period('M')
unique_months = df_bt['month_group'].unique()
monthly_results = []

for m in unique_months:
    df_m = df_bt[df_bt['month_group'] == m].copy()
    equity = MONTHLY_CAPITAL
    active_trade = None
    month_trades_count = 0
    win_count = 0
    
    for idx, row in df_m.iterrows():
        # 1. Manage active trade
        if active_trade:
            t = active_trade
            t['bars'] += 1
            closed = False; reason = ""; exit_price = row['close']
            
            # SL/TP Logic
            if t['side'] == 'LONG':
                if row['low'] <= t['sl']: exit_price = t['sl']; closed=True; reason="SL"
                elif row['high'] >= t['tp']: exit_price = t['tp']; closed=True; reason="TP"
            else: # SHORT
                if row['high'] >= t['sl']: exit_price = t['sl']; closed=True; reason="SL"
                elif row['low'] <= t['tp']: exit_price = t['tp']; closed=True; reason="TP"
            
            # Other exit logic
            if not closed:
                if (t['side'] == 'LONG' and row['prob_short'] > 0.65) or \
                   (t['side'] == 'SHORT' and row['prob_long'] > 0.65):
                    closed = True; reason="AI"
            
            if not closed and t['bars'] > 32:
                closed = True; reason="Time"

            # Close trade at month's end if still open
            if not closed and idx == df_m.index[-1]:
                closed = True; reason="EndMonth"

            if closed:
                pnl_multiplier = 1 if t['side'] == 'LONG' else -1
                gross_pnl = (exit_price - t['entry']) * t['size_asset'] * pnl_multiplier
                
                entry_cost = t['size_usd'] * TRADING_FEE
                exit_cost = (t['size_asset'] * exit_price) * TRADING_FEE
                total_fees = entry_cost + exit_cost
                
                net_pnl = gross_pnl - total_fees
                
                equity += net_pnl
                month_trades_count += 1
                if net_pnl > 0: win_count += 1
                
                active_trade = None
        
        if equity <= 0: break

        # 2. New entry logic
        if not active_trade and not row['is_sideways']:
            is_long = row['prob_long'] > bt_thresh_long and row['prob_short'] < 0.5
            is_short = row['prob_short'] > bt_thresh_short and row['prob_long'] < 0.5
            
            trade_side = None
            if is_long: trade_side = 'LONG'
            if is_short: trade_side = 'SHORT'
            
            if trade_side:
                entry_price = row['close']
                risk_amt = equity * RISK_PER_TRADE
                
                if trade_side == 'LONG':
                    sl_price = entry_price - (row['atr'] * STOP_LOSS_ATR)
                    tp_price = entry_price + (row['atr'] * TAKE_PROFIT_ATR)
                else: # SHORT
                    sl_price = entry_price + (row['atr'] * STOP_LOSS_ATR)
                    tp_price = entry_price - (row['atr'] * TAKE_PROFIT_ATR)
                
                sl_distance = abs(entry_price - sl_price)
                if sl_distance == 0: continue
                
                position_size_asset = risk_amt / sl_distance
                position_size_usd = position_size_asset * entry_price
                capital_required = position_size_usd / LEVERAGE
                
                if capital_required > equity: continue

                active_trade = {
                    'side': trade_side, 'entry': entry_price, 'sl': sl_price, 'tp': tp_price,
                    'bars': 0, 'size_asset': position_size_asset, 'size_usd': position_size_usd
                }

    net_profit = equity - MONTHLY_CAPITAL
    roi = (net_profit / MONTHLY_CAPITAL) * 100
    win_rate = (win_count / month_trades_count * 100) if month_trades_count > 0 else 0
    
    monthly_results.append({
        'Month': str(m),
        'Trades': month_trades_count,
        'WinRate': win_rate,
        'End_Equity': equity,
        'Net_Profit': net_profit,
        'ROI': roi
    })

results_df = pd.DataFrame(monthly_results)

print(f"\n📊 MONTHLY TRADING RESULTS TABLE (BTCUSDT) - HIGH-FIDELITY MODEL")
print(f"   Starting capital each month: ${MONTHLY_CAPITAL:.2f}, Risk/trade: {RISK_PER_TRADE*100}%, Leverage: {LEVERAGE}x")
print("-" * 85)
print(f"{'Month':<10} | {'Trades':<8} | {'WinRate':<8} | {'End Equity($)':<15} | {'Net P/L($)':<12} | {'ROI(%)':<8}")
print("-" * 85)

total_net_profit = 0
for i, row in results_df.iterrows():
    color = CLR.GREEN if row['Net_Profit'] > 0 else CLR.RED
    reset = CLR.END
    print(f"{row['Month']:<10} | {row['Trades']:<8} | {row['WinRate']:>6.1f}% | {row['End_Equity']:>15.2f} | {color}{row['Net_Profit']:>12.2f}{reset} | {color}{row['ROI']:>7.2f}%{reset}")
    if not np.isnan(row['Net_Profit']):
        total_net_profit += row['Net_Profit']

print("-" * 85)
avg_roi = results_df['ROI'].mean()
avg_monthly_profit = results_df['Net_Profit'].mean()
print(f"💰 TOTAL PROFIT AFTER 1 YEAR (withdrawing profits monthly): ${total_net_profit:,.2f}")
print(f"📈 Average Monthly ROI: {avg_roi:.2f}%")
print(f"💵 Average Monthly Profit: ${avg_monthly_profit:.2f} / month")

# Plotting monthly results chart
plt.figure(figsize=(12, 6))
colors = ['green' if x > 0 else 'red' for x in results_df['Net_Profit']]
plt.bar(results_df['Month'], results_df['Net_Profit'], color=colors, alpha=0.8)
plt.axhline(0, color='black', linewidth=1)
plt.title(f'Monthly Net Profit (Capital Reset to ${MONTHLY_CAPITAL}) - Total Profit: ${total_net_profit:.2f}', fontsize=14)
plt.ylabel('Profit/Loss ($)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# =============================================================================
# BLOCK 6: MODEL ROBUSTNESS CHECK (WALK-FORWARD VALIDATION)
# Purpose: Evaluate the model's performance across different market periods 
# to check for stability, rather than just good performance on a specific test set.
# =============================================================================
print("\n" + "="*60)
print(" R RUNNING WALK-FORWARD VALIDATION TEST")
print("="*60)

from sklearn.model_selection import TimeSeriesSplit

# Use the entire filtered dataset for the check
X_wf = X
y_long_wf = y_long
y_short_wf = y_short

# Split the data into 5 folds chronologically
# Fold 1: Train [block 1], Test [block 2]
# Fold 2: Train [block 1, 2], Test [block 3]
# ...
N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

walk_forward_results = []
fold_count = 0

# Reuse the hyperparameters optimized in the previous step
# We will retrain the model on each fold using this same set of parameters
model_long_wf = lgb.LGBMClassifier(**final_params_long)
model_short_wf = lgb.LGBMClassifier(**final_params_short)

for train_index, test_index in tscv.split(X_wf):
    fold_count += 1
    print(f"\n--- FOLD {fold_count}/{N_SPLITS} ---")

    # Split data for the current fold
    X_train_wf, X_test_wf = X_wf.iloc[train_index], X_wf.iloc[test_index]
    y_long_train_wf, y_long_test_wf = y_long_wf.iloc[train_index], y_long_wf.iloc[test_index]
    y_short_train_wf, y_short_test_wf = y_short_wf.iloc[train_index], y_short_wf.iloc[test_index]

    train_start, train_end = X_train_wf.index.min(), X_train_wf.index.max()
    test_start, test_end = X_test_wf.index.min(), X_test_wf.index.max()
    print(f"   Train period: {train_start} -> {train_end} ({len(X_train_wf)} samples)")
    print(f"   Test period:  {test_start} -> {test_end} ({len(X_test_wf)} samples)")

    # Retrain models on this fold's training data
    print("   ⏳ Re-training models for this fold...")
    model_long_wf.fit(X_train_wf, y_long_train_wf)
    model_short_wf.fit(X_train_wf, y_short_train_wf)

    # Evaluate on this fold's test data
    pred_long = model_long_wf.predict_proba(X_test_wf)[:, 1]
    pred_short = model_short_wf.predict_proba(X_test_wf)[:, 1]

    auc_long = roc_auc_score(y_long_test_wf, pred_long)
    auc_short = roc_auc_score(y_short_test_wf, pred_short)
    
    print(f"   ✅ Fold {fold_count} Results -> Long AUC: {auc_long:.4f} | Short AUC: {auc_short:.4f}")

    walk_forward_results.append({
        'Fold': fold_count,
        'Test_Start': test_start.strftime('%Y-%m-%d'),
        'Test_End': test_end.strftime('%Y-%m-%d'),
        'AUC_Long': auc_long,
        'AUC_Short': auc_short
    })

# --- Summarize and Analyze Walk-Forward Results ---
print("\n" + "="*60)
print("📊 WALK-FORWARD VALIDATION SUMMARY")
print("="*60)
results_wf_df = pd.DataFrame(walk_forward_results)
print(results_wf_df.to_string(index=False, float_format="%.4f"))

avg_auc_long = results_wf_df['AUC_Long'].mean()
std_auc_long = results_wf_df['AUC_Long'].std()
avg_auc_short = results_wf_df['AUC_Short'].mean()
std_auc_short = results_wf_df['AUC_Short'].std()

print("\n--- Overall Performance ---")
print(f"📈 AVERAGE Long AUC:  {avg_auc_long:.4f} (Std Dev: {std_auc_long:.4f})")
print(f"📉 AVERAGE Short AUC: {avg_auc_short:.4f} (Std Dev: {std_auc_short:.4f})")

print("\n--- Interpretation ---")
if std_auc_long > 0.1 or std_auc_short > 0.1:
    print("   ⚠️ WARNING: The model's performance is very unstable across different market periods (high Std Dev).")
    print("   -> The model may not be reliable in a live trading environment.")
else:
    print("   ✅ GOOD: The model's performance is relatively stable across periods (low Std Dev).")

if avg_auc_long < 0.65 or avg_auc_short < 0.65:
    print("   ❌ CONCERNING: The model's average performance is quite low. The model or features need improvement.")
else:
    print("   ✅ GOOD: The model's average performance is at an acceptable/good level.")