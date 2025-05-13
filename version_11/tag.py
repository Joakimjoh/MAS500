"""Third-Party Libraries"""
from sklearn.linear_model import LinearRegression
import numpy as np
import csv
import os

"""Internal Modules"""
from camera import Camera

class Tag:
    """Handles RealSense camera initialization and continuous frame fetching."""
    def __init__(self,
        camera: Camera,
        side: str
    ) -> None:
        self.camera = camera

        self.orientation = camera.get_orientation(side)

        self.model = self.get_error_model()

    # Function to read data from the CSV files
    def get_region_data(self):
        region_data = []

        if not os.path.exists("region.csv"):
            self.camera.create_sample_region()

        # Read region.csv
        with open("region.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                x, y, depth = int(row[0]), int(row[1]), float(row[2])
                region_data.append((x, y, depth))

        return region_data

    def get_linear_reg_error(self, point):
        x, y = point
        input_array = np.array([x, y]).flatten().reshape(1, -1)  # Flatten and reshape to 2D
        
        # Ensure no dimensionality error
        z_pred = self.model.predict(input_array)  # Correct way to use predict

        return z_pred
    
    def adjust_error(self, point):
        error = self.get_linear_reg_error((point[0], point[1]))
        point[2] -= error
        if point[2] < 0.000:
            point[2] = 0.000
        return point
    
    def reverse_adjust_error(self, point):
        """Reverse the Z-axis error adjustment by adding the predicted depth error back."""
        error = self.get_linear_reg_error((point[0], point[1]))
        point[2] += error
        return point

    # LINEAR REGRESSION FITTING
    def linear_reg(self, x_values, y_values, z_values):
        # Combine x and y into a single array of shape (n_samples, n_features)
        x_values = np.array(x_values).flatten()
        y_values = np.array(y_values).flatten()
        z_values = np.array(z_values).flatten()
        X = np.column_stack((x_values, y_values))

        # Create and fit the model
        model = LinearRegression()
        model.fit(X, z_values)

        return model

    def get_error_model(self):
        # Read CSV data
        region_data = self.get_region_data()

        x_values = []
        y_values = []
        z_values = []

        # Process the pixels in the region
        for x, y, depth in region_data:
            if depth > 0:
                point_tag = self.camera.pixel_to_coordsystem(self, (x, y, depth))
                x_values.append(point_tag[0])
                y_values.append(point_tag[1])
                z_values.append(point_tag[2])

        # Combine x and y into a single array of shape (n_samples, n_features)
        x_values = np.array(x_values).flatten()
        y_values = np.array(y_values).flatten()
        z_values = np.array(z_values).flatten()
        X = np.column_stack((x_values, y_values))

        # Create and fit the model
        model = LinearRegression()
        model.fit(X, z_values)
        
        return model
