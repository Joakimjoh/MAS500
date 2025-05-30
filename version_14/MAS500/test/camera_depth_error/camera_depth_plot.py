import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("/home/student/Documents/MAS500/result_data/depth_data_without_1cm_error.csv")
df2 = pd.read_csv("/home/student/Documents/MAS500/result_data/depth_data_with_1cm_error.csv")

# Filter for py = 120
df_py = df[df["py"] == 120].copy()
df_py2 = df2[df2["py"] == 120].copy()

# Convert string values to float
x = df_py['x (m)'].str.strip('[]').astype(float).values
z = df_py['z (m)'].str.strip('[]').astype(float).values

x2 = df_py2['x (m)'].str.strip('[]').astype(float).values
z2 = df_py2['z (m)'].str.strip('[]').astype(float).values

# Sort values by x
sorted_idx = np.argsort(x)
x = x[sorted_idx]
z = z[sorted_idx]

sorted_idx2 = np.argsort(x2)
x2 = x2[sorted_idx2]
z2 = z2[sorted_idx2]

# Line plot with workspace surface
plt.figure(figsize=(10, 6))
plt.plot(x, z, color='blue', label='Without 1cm Error', linewidth=2)
plt.plot(x2, z2, color='red', label='With 1cm Error', linewidth=2)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, label='Workspace Surface')
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('z (m)', fontsize=14)
plt.title('x vs z Line Plot (py = 120)', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.show()
