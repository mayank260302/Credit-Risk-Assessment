import pandas as pd
import numpy as np
import os

def calculate_expected_loss(pd_values):
    os.makedirs("outputs", exist_ok=True)

    # Simulated LGD (industry realistic: 20%–60%)
    lgd = np.random.uniform(0.2, 0.6, size=len(pd_values))

    # Simulated EAD (loan amount)
    ead = np.random.uniform(50000, 500000, size=len(pd_values))

    expected_loss = pd_values * lgd * ead

    el_df = pd.DataFrame({
        "PD": pd_values,
        "LGD": lgd,
        "EAD": ead,
        "Expected_Loss": expected_loss
    })

    el_df.to_csv("outputs/expected_loss.csv", index=False)

    print("expected_loss.csv generated")

    return el_df
