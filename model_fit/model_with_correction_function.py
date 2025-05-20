import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor
from scipy.spatial.distance import cdist

def build_model_from_csv(csv_path='region.csv'):
    global scaler_xy, scaler_z, poly, centers, model

    df = pd.read_csv(csv_path)
    x = df['x'].astype(float).values
    y = df['y'].astype(float).values
    z = df['z'].astype(float).values

    # Normalize features
    scaler_xy = StandardScaler()
    XY = scaler_xy.fit_transform(np.vstack([x, y]).T)

    scaler_z = StandardScaler()
    z_norm = scaler_z.fit_transform(z.reshape(-1, 1)).ravel()

    # Polynomial features
    poly = PolynomialFeatures(degree=4, include_bias=False)
    X_poly = poly.fit_transform(XY)

    # RBF features
    def rbf_features(X, centers, sigma):
        dist = cdist(X, centers, 'sqeuclidean')
        return np.exp(-dist / (2 * sigma**2))

    q_low, q_high = 0.01, 0.99
    xq = np.quantile(XY[:, 0], [q_low, q_high])
    yq = np.quantile(XY[:, 1], [q_low, q_high])
    x_centers = np.linspace(xq[0], xq[1], 7)
    y_centers = np.linspace(yq[0], yq[1], 7)
    xc, yc = np.meshgrid(x_centers, y_centers)
    centers = np.vstack([xc.ravel(), yc.ravel()]).T
    X_rbf = rbf_features(XY, centers, sigma=0.3)

    X_full = np.hstack([X_poly, X_rbf])

    X_train, X_test, z_train, z_test = train_test_split(X_full, z_norm, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.14,
        max_depth=14,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.set_params(early_stopping_rounds=20)
    model.fit(X_train, z_train, eval_set=[(X_test, z_test)], verbose=False)

    return model

def correct_z_error(x, y, z_measured):
    XY_new = scaler_xy.transform([[x, y]])
    poly_feats = poly.transform(XY_new)

    dist = cdist(XY_new, centers, 'sqeuclidean')
    rbf_feats = np.exp(-dist / (2 * 0.3**2))

    X_input = np.hstack([poly_feats, rbf_feats])
    z_error_est = scaler_z.inverse_transform(model.predict(X_input).reshape(-1, 1)).ravel()[0]
    z_corrected = z_measured - z_error_est
    return z_corrected
