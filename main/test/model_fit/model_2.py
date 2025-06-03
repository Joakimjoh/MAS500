import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler

# === Load and clean data ===
df = pd.read_csv('/home/student/Documents/MAS500/model_fit/depth_data_without_error_model7.csv')
x = df['x (m)'].str.strip('[]').astype(float).values
y = df['y (m)'].str.strip('[]').astype(float).values
z = df['z (m)'].str.strip('[]').astype(float).values

# === Normalize features ===
scaler_xy = StandardScaler()
XY = scaler_xy.fit_transform(np.vstack([x, y]).T)

scaler_z = StandardScaler()
z_norm = scaler_z.fit_transform(z.reshape(-1, 1)).ravel()

# === Polynomial features (4th-order) ===
poly = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly.fit_transform(XY)

# === Fast RBF features using RBFSampler ===
rbf_sigma = 0.3
gamma = 1.0 / (2 * rbf_sigma ** 2)
rbf_sampler = RBFSampler(gamma=gamma, n_components=49, random_state=42)
X_rbf = rbf_sampler.fit_transform(XY)

# === Combine poly + RBF features ===
X_full = np.hstack([X_poly, X_rbf])

# === Train/test split ===
X_train, X_test, z_train, z_test = train_test_split(X_full, z_norm, test_size=0.2, random_state=42)

# === Fit XGBoost ===
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

# === Predict and evaluate ===
z_pred_train = scaler_z.inverse_transform(model.predict(X_train).reshape(-1, 1)).ravel()
z_pred_test = scaler_z.inverse_transform(model.predict(X_test).reshape(-1, 1)).ravel()
z_test_true = scaler_z.inverse_transform(z_test.reshape(-1, 1)).ravel()

train_rmse = np.sqrt(mean_squared_error(scaler_z.inverse_transform(z_train.reshape(-1, 1)), z_pred_train))
test_rmse = np.sqrt(mean_squared_error(z_test_true, z_pred_test))

print(f"Train RMSE: {train_rmse:.6f}")
print(f"Test  RMSE: {test_rmse:.6f}")

# === Residual plot ===
residuals = z_test_true - z_pred_test
plt.figure(figsize=(6,5))
plt.hist(residuals, bins=40, edgecolor='k')
plt.title('Test Residual Distribution')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Predict on full data for visualization ===
z_fit_all = scaler_z.inverse_transform(model.predict(X_full).reshape(-1, 1)).ravel()

# === Compute and display test set averages ===
mean_actual_z = np.mean(np.abs(z_test_true))
mean_predicted_z = np.mean(np.abs(z_pred_test))
mean_residual = np.mean(np.abs(z_test_true - z_pred_test))

print("\n=== Test Set Averages ===")
print(f"Mean |Actual z|:     {mean_actual_z:.6f}")
print(f"Mean |Predicted z|:  {mean_predicted_z:.6f}")
print(f"Mean |Residual|:     {mean_residual:.6f}")

# === Max residual statistics ===
z_test_residuals = np.abs(z_test_true - z_pred_test)
max_idx = np.argmax(z_test_residuals)
max_actual = z_test_true[max_idx]
max_predicted = z_pred_test[max_idx]
max_residual = z_test_residuals[max_idx]

print("\n=== Max Residual ===")
print(f"Max Actual z:     {max_actual:.6f}")
print(f"Max Predicted z:  {max_predicted:.6f}")
print(f"Max |Residual|:    {max_residual:.6f}")
