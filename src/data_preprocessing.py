import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from .config import DATA_PATH, TARGET

def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)

    # Drop ID column
    df.drop(columns=["SK_ID_CURR"], inplace=True)

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # Separate numeric and categorical
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # Impute
    X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
    X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])

    # One-hot encode
    X = pd.get_dummies(X, drop_first=True)

    return train_test_split(X, y, test_size=0.2, random_state=42)
    
def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["SK_ID_CURR"])
    return df
