import os
import pandas as pd

os.makedirs("outputs", exist_ok=True)

# ================================
# IMPORTS
# ================================
from src.data_preprocessing import load_and_clean_data, load_raw_data
from src.pd_model import train_pd_model
from src.scorecard import generate_scorecard
from src.expected_loss import calculate_expected_loss
from src.woe_iv import calculate_woe_iv, apply_woe

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

print("Pipeline started")

# ================================
# STEP 1: LOAD RAW DATA
# ================================
raw_df = load_raw_data()

# ================================
# STEP 2: WoE + IV CALCULATION
# ================================
numeric_cols = raw_df.select_dtypes(include=["int64", "float64"]).columns
numeric_cols = [c for c in numeric_cols if c != "TARGET"]

iv_summary = []

print("\nCalculating IV for numeric features...\n")

for col in numeric_cols[:10]:  # limit to 10 features
    _, iv = calculate_woe_iv(raw_df, col, "TARGET")
    iv_summary.append({"feature": col, "IV": iv})

iv_df = pd.DataFrame(iv_summary).sort_values(by="IV", ascending=False)

print("Information Value (IV) Summary:")
print(iv_df)

# ================================
# STEP 3: SELECT TOP IV FEATURES
# ================================
top_features = iv_df[iv_df["IV"] > 0.1]["feature"].tolist()

print("\nTop IV Features Selected:")
print(top_features)

# ================================
# STEP 4: CREATE WoE DATASET
# ================================
woe_df = pd.DataFrame()
woe_df["TARGET"] = raw_df["TARGET"]

for feature in top_features:
    woe_df[feature + "_WOE"] = apply_woe(raw_df, feature, "TARGET")

print("\nWoE dataset shape:", woe_df.shape)

# ================================
# STEP 5: WoE-BASED PD MODEL
# ================================
X = woe_df.drop(columns=["TARGET"])
y = woe_df["TARGET"]

X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X, y, test_size=0.2, random_state=42
)

woe_model = LogisticRegression(max_iter=2000)
woe_model.fit(X_train_w, y_train_w)

woe_preds = woe_model.predict_proba(X_test_w)[:, 1]
woe_auc = roc_auc_score(y_test_w, woe_preds)

print(f"\nWoE-based PD Model AUC: {woe_auc:.4f}")

# ================================
# STEP 6: STANDARD PD MODEL
# ================================
X_train, X_test, y_train, y_test = load_and_clean_data()
print("\nProcessed data loaded")

pd_model = train_pd_model(X_train, X_test, y_train, y_test)
print("Standard PD model trained")

pd_preds = pd_model.predict_proba(X_test)[:, 1]

# ================================
# STEP 7: SCORECARD
# ================================
generate_scorecard(pd_preds)

# ================================
# STEP 8: EXPECTED LOSS
# ================================
calculate_expected_loss(pd_preds)

print("\n Pipeline finished successfully")
