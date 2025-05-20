from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor
from scipy.spatial.distance import cdist
import numpy as np
import joblib
import os
import csv

from camera import Camera  # Assuming the camera module is present

class Tag:
    def __init__(self, camera: Camera, side: str) -> None:
        self.camera = camera
        self.side = side
        self.orientation = camera.get_orientation(side)

        while self.orientation is None:
            camera.frame.text = f"Error: AprilTag(s) not found. Press Enter to try again"
            while camera.key != 13:
                pass
            self.orientation = camera.get_orientation(side)

        self.model = self.get_error_model()

    def get_region_data(self):
        region_data = []
        if not os.path.exists("region.csv"):
            self.camera.create_sample_region()

        with open("region.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                x, y, depth = int(row[0]), int(row[1]), float(row[2])
                region_data.append((x, y, depth))
        return region_data

    def get_error_model(self):
        model_path = f"error_model_{self.side}.joblib"
        if os.path.exists(model_path):
            return joblib.load(model_path)

        region_data = self.get_region_data()
        x_values, y_values, z_values = [], [], []

        for x, y, depth in region_data:
            if depth > 0:
                point_tag = self.camera.pixel_to_coordsystem(self, (x, y, depth))
                x_values.append(point_tag[0])
                y_values.append(point_tag[1])
                z_values.append(point_tag[2])

        if not x_values:
            raise ValueError("No valid region data for training error model.")

        XY = np.vstack([x_values, y_values]).T

        self.scaler_xy = StandardScaler().fit(XY)
        XY_scaled = self.scaler_xy.transform(XY)

        self.scaler_z = StandardScaler().fit(np.array(z_values).reshape(-1, 1))
        z_norm = self.scaler_z.transform(np.array(z_values).reshape(-1, 1)).ravel()

        self.poly = PolynomialFeatures(degree=4, include_bias=False)
        X_poly = self.poly.fit_transform(XY_scaled)

        # RBF features
        q_low, q_high = 0.01, 0.99
        xq = np.quantile(XY_scaled[:, 0], [q_low, q_high])
        yq = np.quantile(XY_scaled[:, 1], [q_low, q_high])
        x_centers = np.linspace(xq[0], xq[1], 7)
        y_centers = np.linspace(yq[0], yq[1], 7)
        xc, yc = np.meshgrid(x_centers, y_centers)
        self.rbf_centers = np.vstack([xc.ravel(), yc.ravel()]).T

        def rbf_features(X, centers, sigma=0.3):
            dist = cdist(X, centers, 'sqeuclidean')
            return np.exp(-dist / (2 * sigma ** 2))

        self.X_rbf = rbf_features(XY_scaled, self.rbf_centers)
        self.X_train = np.hstack([X_poly, self.X_rbf])
        self.z_train = z_norm

        self.model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.14,
            max_depth=14,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.model.set_params(early_stopping_rounds=20)
        self.model.fit(self.X_train, self.z_train)

        joblib.dump(self, model_path)
        return self

    def get_linear_reg_error(self, point_xy):
        input_xy = self.scaler_xy.transform([point_xy])
        poly_feat = self.poly.transform(input_xy)
        rbf_feat = np.exp(-cdist(input_xy, self.rbf_centers, 'sqeuclidean') / (2 * 0.3**2))
        X_input = np.hstack([poly_feat, rbf_feat])
        z_error = self.scaler_z.inverse_transform(self.model.predict(X_input).reshape(-1, 1)).ravel()[0]
        return z_error

    def adjust_error(self, point):
        if point is None or any(np.isnan(point)) or any(np.isinf(point)):
            return point
        error = self.get_linear_reg_error((point[0], point[1]))
        point[2] -= error
        return point

    def reverse_adjust_error(self, point):
        error = self.get_linear_reg_error((point[0], point[1]))
        point[2] += error
        return point
