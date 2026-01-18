import pandas as pd
import numpy as np
import os

def generate_scorecard(pd_values):
    print("👉 generate_scorecard() called")

    os.makedirs("outputs", exist_ok=True)

    score = 600 - (50 * np.log(pd_values / (1 - pd_values)))

    scorecard = pd.DataFrame({
        "PD": pd_values,
        "Credit_Score": score
    })

    scorecard.to_csv("outputs/scorecard.csv", index=False)

    print("✅ scorecard.csv generated")
