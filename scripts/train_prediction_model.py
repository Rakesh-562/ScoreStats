"""
scripts/train_prediction_model.py
 
Run AFTER build_training_data.py:
    python scripts/train_prediction_model.py
 
What it does:
    - Reads data/training_data.csv
    - Trains Model A: predicts final innings score
    - Trains Model B: predicts win probability (2nd innings)
    - Saves both models to models_ml/ folder
 
You do NOT need to understand the math.
Just know: more data = better predictions.
"""
 
import os
import sys
import json
 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 
import pandas as pd
import numpy as np
import joblib
 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, accuracy_score
 
# ─────────────────────────────────────────────────────────────────────────────
# FEATURE LISTS
#
# IMPORTANT: These lists must match EXACTLY what build_training_data.py saved.
# If you add a column to the CSV, add it here too.
# If you rename a column in the CSV, rename it here too.
# ─────────────────────────────────────────────────────────────────────────────
 
# Features used by the score predictor (works for both innings)
BASE_FEATURES = [
    'runs_so_far',
    'wickets_fallen',
    'balls_bowled',
    'balls_remaining',
    'overs_completed',
    'current_run_rate',
    'striker_runs',
    'striker_balls',
    'striker_strike_rate',
    'striker_fours',
    'striker_sixes',
    'bowler_economy',
    'bowler_wickets',
    'bowler_dots',
    'over_limit',
]
 
# Extra features for the win predictor (only meaningful in 2nd innings)
WIN_EXTRA_FEATURES = [
    'required_run_rate',
    'runs_to_target',
    'target',
]
 
WIN_FEATURES = BASE_FEATURES + WIN_EXTRA_FEATURES
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MODEL A — Score Predictor
# ─────────────────────────────────────────────────────────────────────────────
 
def train_score_predictor(df):
    print("\n" + "-" * 50)
    print("TRAINING MODEL A — Score Predictor")
    print("-" * 50)
 
    # Keep only the columns we need and drop rows with missing values
    df_clean = df[BASE_FEATURES + ['final_score']].dropna()
    print(f"Rows used for training: {len(df_clean)}")
 
    if len(df_clean) < 20:
        print("Not enough data. Record more matches first.")
        return None
 
    X = df_clean[BASE_FEATURES].values
    y = df_clean['final_score'].values
 
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 
    # Build the Random Forest
    # n_estimators=200    → 200 decision trees vote together
    # max_depth=12        → trees can't go deeper than 12 splits
    # min_samples_split=5 → a branch needs 5+ samples to split
    # min_samples_leaf=3  → leaf must have 3+ samples
    # These numbers prevent the model from "memorising" your small dataset
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
 
    model.fit(X_train, y_train)
 
    # How accurate is it?
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
 
    print(f"\nMean Absolute Error: {mae:.1f} runs")
    print(f"(prediction is off by ~{mae:.0f} runs on average)")
 
    # Cross-validation is more reliable than a single test
    cv = cross_val_score(
        model, X, y,
        cv=min(5, len(df_clean) // 10),
        scoring='neg_mean_absolute_error'
    )
    print(f"Cross-val MAE: {-cv.mean():.1f} runs ± {cv.std():.1f}")
 
    # Which features matter most?
    importances = sorted(
        zip(BASE_FEATURES, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    print("\nFeature importance (top 8):")
    for name, imp in importances[:8]:
        bar = "█" * int(imp * 50)
        print(f"  {name:<25} {bar}  {imp:.3f}")
 
    return model
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MODEL B — Win Predictor
# ─────────────────────────────────────────────────────────────────────────────
 
def train_win_predictor(df):
    print("\n" + "-" * 50)
    print("TRAINING MODEL B — Win Predictor (2nd innings)")
    print("-" * 50)
 
    # Only 2nd innings has a meaningful target/required-run-rate
    df_2nd = df[df['innings_number'] == 2].copy()
 
    print(f"2nd innings rows available: {len(df_2nd)}")
 
    if len(df_2nd) < 30:
        print("Not enough 2nd innings data. Need at least 15 completed 2nd innings.")
        print("Skipping win predictor training.")
        return None
 
    # Win label: chasing team won if their final score >= target
    df_2nd['won'] = (df_2nd['final_score'] >= df_2nd['target']).astype(int)
 
    df_clean = df_2nd[WIN_FEATURES + ['won']].dropna()
    print(f"Rows after cleaning: {len(df_clean)}")
    print(f"Win rate in data: {df_clean['won'].mean():.1%}")
 
    X = df_clean[WIN_FEATURES].values
    y = df_clean['won'].values
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    # class_weight='balanced' is important when you have unequal
    # numbers of wins vs losses
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
 
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
 
    cv = cross_val_score(
        model, X, y,
        cv=min(5, len(df_clean) // 10),
        scoring='accuracy'
    )
 
    print(f"\nTest accuracy:  {acc:.1%}")
    print(f"Cross-val acc:  {cv.mean():.1%} ± {cv.std():.1%}")
    print(f"(The paper got ~90% with thousands of IPL matches)")
 
    importances = sorted(
        zip(WIN_FEATURES, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    print("\nFeature importance (top 8):")
    for name, imp in importances[:8]:
        bar = "█" * int(imp * 50)
        print(f"  {name:<25} {bar}  {imp:.3f}")
 
    return model
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
 
def save_everything(score_model, win_model):
    os.makedirs('models_ml', exist_ok=True)
 
    if score_model:
        joblib.dump(score_model, 'models_ml/score_predictor.pkl')
        print("\nSaved → models_ml/score_predictor.pkl")
 
    if win_model:
        joblib.dump(win_model, 'models_ml/win_predictor.pkl')
        print("Saved → models_ml/win_predictor.pkl")
 
    # Save feature list so prediction_service.py uses the exact same order
    meta = {
        'base_features': BASE_FEATURES,
        'win_features':  WIN_FEATURES,
    }
    with open('models_ml/feature_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print("Saved → models_ml/feature_meta.json")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == '__main__':
    data_path = 'data/training_data.csv'
 
    if not os.path.exists(data_path):
        print(f"\nERROR: {data_path} not found.")
        print("Run this first:  python scripts/build_training_data.py")
        sys.exit(1)
 
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")
 
    score_model = train_score_predictor(df)
    win_model   = train_win_predictor(df)
 
    save_everything(score_model, win_model)
 
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("Models are saved. Your API will now serve predictions.")
    print("=" * 60)
    print("\nTo retrain after more matches:")
    print("  python scripts/build_training_data.py")
    print("  python scripts/train_prediction_model.py")
 