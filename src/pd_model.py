from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import os

def train_pd_model(X_train, X_test, y_train, y_test):
    os.makedirs("outputs", exist_ok=True)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=3000,
            solver="lbfgs"
        ))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    print(f"PD Model AUC: {auc:.4f}")

    pd.DataFrame({"PD": preds}).to_csv(
        "outputs/pd_predictions.csv", index=False
    )

    return pipeline
