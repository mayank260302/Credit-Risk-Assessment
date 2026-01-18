import pandas as pd
import numpy as np

def calculate_woe_iv(df, feature, target, bins=10):
    df = df[[feature, target]].copy()

    # Bin the feature
    df["bin"] = pd.qcut(df[feature], q=bins, duplicates="drop")

    grouped = df.groupby("bin")

    stats = grouped[target].agg(["count", "sum"])
    stats.columns = ["total", "bad"]
    stats["good"] = stats["total"] - stats["bad"]

    # Avoid division by zero
    stats["bad"] = stats["bad"].replace(0, 0.5)
    stats["good"] = stats["good"].replace(0, 0.5)

    stats["bad_rate"] = stats["bad"] / stats["bad"].sum()
    stats["good_rate"] = stats["good"] / stats["good"].sum()

    stats["woe"] = np.log(stats["good_rate"] / stats["bad_rate"])
    stats["iv"] = (stats["good_rate"] - stats["bad_rate"]) * stats["woe"]

    iv = stats["iv"].sum()

    return stats, iv

def apply_woe(df, feature, target, bins=10):
    df_temp = df[[feature, target]].copy()

    # Bin feature
    df_temp["bin"] = pd.qcut(df_temp[feature], q=bins, duplicates="drop")

    grouped = df_temp.groupby("bin")

    stats = grouped[target].agg(["count", "sum"])
    stats.columns = ["total", "bad"]
    stats["good"] = stats["total"] - stats["bad"]

    stats["bad"] = stats["bad"].replace(0, 0.5)
    stats["good"] = stats["good"].replace(0, 0.5)

    stats["bad_rate"] = stats["bad"] / stats["bad"].sum()
    stats["good_rate"] = stats["good"] / stats["good"].sum()

    stats["woe"] = np.log(stats["good_rate"] / stats["bad_rate"])

    # Map WoE back
    woe_map = stats["woe"].to_dict()

    df_temp["woe"] = df_temp["bin"].map(woe_map)

    return df_temp["woe"]
    
