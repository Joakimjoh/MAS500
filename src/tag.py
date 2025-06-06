import numpy as np
import csv
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor
from scipy.spatial.distance import cdist
from camera import Camera

class Tag:
    def __init__(self, camera: Camera, side: str) -> None:
        """
        Initialize the Tag object for a specific camera side.
        Retrieves orientation and loads or trains the error correction model.
        """
        self.camera = camera
        self.side = side
        self.orientation = camera.get_orientation(side)

        while self.orientation is None:
            camera.frame.text = "Error: AprilTag(s) not found. Press Enter to try again"
            while camera.key != 13:
                pass
            self.orientation = camera.get_orientation(side)

        camera.frame.text = f"Training error model for {side} side."
        self.load_or_train_model()

    def get_region_data(self):
        """
        Load region sample data from CSV file.
        If file doesn't exist, calls camera to create it.
        """
        if not os.path.exists("region.csv"):
            self.camera.create_sample_region()

        region_data = []
        with open("region.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                x, y, depth = int(row[0]), int(row[1]), float(row[2])
                region_data.append((x, y, depth))
        return region_data

    def load_or_train_model(self):
        """
        Load an existing error correction model or train a new one
        using region sample data and a combination of polynomial and RBF features.
        """
        model_path = f"error_model_{self.side}.joblib"

        if os.path.exists(model_path):
            data = joblib.load(model_path)
            self.model = data['model']
            self.scaler_xy = data['scaler_xy']
            self.scaler_z = data['scaler_z']
            self.poly = data['poly']
            self.rbf_centers = data['rbf_centers']
            self.rbf_sigma = data['rbf_sigma']
            return

        # === Load region data ===
        region_data = self.get_region_data()
        x_vals, y_vals, z_vals = [], [], []

        for x, y, depth in region_data:
            if depth > 0:
                point = self.camera.pixel_to_coordsystem(self, (x, y, depth))
                x_vals.append(point[0])
                y_vals.append(point[1])
                z_vals.append(point[2])

        if len(x_vals) < 10:
            raise ValueError("Not enough valid region data to train the model.")

        # === Preprocess ===
        XY = np.column_stack((x_vals, y_vals))
        self.scaler_xy = StandardScaler().fit(XY)
        XY_scaled = self.scaler_xy.transform(XY)

        self.scaler_z = StandardScaler().fit(np.array(z_vals).reshape(-1, 1))
        z_norm = self.scaler_z.transform(np.array(z_vals).reshape(-1, 1)).ravel()

        # === Features: Polynomial + RBF ===
        self.poly = PolynomialFeatures(degree=4, include_bias=False)
        X_poly = self.poly.fit_transform(XY_scaled)

        self.rbf_sigma = 0.3
        xq = np.quantile(XY_scaled[:, 0], [0.01, 0.99])
        yq = np.quantile(XY_scaled[:, 1], [0.01, 0.99])
        x_centers = np.linspace(xq[0], xq[1], 7)
        y_centers = np.linspace(yq[0], yq[1], 7)
        xc, yc = np.meshgrid(x_centers, y_centers)
        self.rbf_centers = np.vstack([xc.ravel(), yc.ravel()]).T

        def rbf_features(X):
            """
            Generate RBF feature matrix for input X.
            """
            dist = cdist(X, self.rbf_centers, 'sqeuclidean')
            return np.exp(-dist / (2 * self.rbf_sigma ** 2))

        X_rbf = rbf_features(XY_scaled)
        X_full = np.hstack([X_poly, X_rbf])

        # === Train model ===
        self.model = XGBRegressor(
            n_estimators=30,
            learning_rate=0.16,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        X_train, X_val, z_train, z_val = train_test_split(
            X_full, z_norm, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, z_train, eval_set=[(X_val, z_val)], verbose=False)

        # === Save model ===
        joblib.dump({
            'model': self.model,
            'scaler_xy': self.scaler_xy,
            'scaler_z': self.scaler_z,
            'poly': self.poly,
            'rbf_centers': self.rbf_centers,
            'rbf_sigma': self.rbf_sigma
        }, model_path)

    def get_linear_reg_error(self, point_xy):
        """
        Predict the error correction for a 2D input point using the trained model.
        """
        point_xy = np.array(point_xy).reshape(1, -1)
        input_xy = self.scaler_xy.transform(point_xy)
        poly_feat = self.poly.transform(input_xy)
        dist = cdist(input_xy, self.rbf_centers, 'sqeuclidean')
        rbf_feat = np.exp(-dist / (2 * self.rbf_sigma ** 2))
        X_input = np.hstack([poly_feat, rbf_feat])
        z_norm = self.model.predict(X_input).reshape(-1, 1)
        return self.scaler_z.inverse_transform(z_norm).ravel()[0]

    def adjust_error(self, point):
        """
        Apply learned correction to a 3D point by adjusting the z-coordinate.
        """
        if point is None or not np.isfinite(point).all():
            return point
        point[2] -= self.get_linear_reg_error(point[:2])
        return point

    def reverse_adjust_error(self, point):
        """
        Undo error correction from a previously adjusted 3D point.
        """
        if point is None or not np.isfinite(point).all():
            return point
        point[2] += self.get_linear_reg_error(point[:2])
        return point
    
    def batch_adjust_error(self, points):
        """
        Apply error correction to an array of 3D points (Nx3).
        Only adjusts valid (finite) points.
        """
        points = np.asarray(points)

        # === Validate input ===
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input must be a (N, 3) array of 3D points.")

        output = points.copy()

        # === Check for valid (finite) rows ===
        valid_mask = np.isfinite(points).all(axis=1)
        if not np.any(valid_mask):
            return output  # No valid points to adjust

        # === Process only valid points ===
        valid_points = points[valid_mask]
        xy = valid_points[:, :2]

        # Scale and transform
        input_xy = self.scaler_xy.transform(xy)
        poly_feat = self.poly.transform(input_xy)
        dist = cdist(input_xy, self.rbf_centers, 'sqeuclidean')
        rbf_feat = np.exp(-dist / (2 * self.rbf_sigma ** 2))
        X_input = np.hstack([poly_feat, rbf_feat])

        # Predict and correct z
        z_norm = self.model.predict(X_input).reshape(-1, 1)
        z_corr = self.scaler_z.inverse_transform(z_norm).ravel()

        output[valid_mask, 2] -= z_corr
        return output
