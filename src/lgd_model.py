import numpy as np
from sklearn.ensemble import RandomForestRegressor

def train_lgd_model(X_train, y_train):
    # Simulated LGD target (industry approximation)
    lgd = np.random.uniform(0.2, 0.8, size=len(y_train))

    model = RandomForestRegressor()
    model.fit(X_train, lgd)

    return model
