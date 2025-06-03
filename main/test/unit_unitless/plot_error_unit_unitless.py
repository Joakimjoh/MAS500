import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load and clean data ===
def clean_dataframe(path):
    df = pd.read_csv(path)
    for axis in ["x (m)", "y (m)", "z (m)"]:
        df[axis] = df[axis].apply(lambda s: abs(float(s.strip("[]"))) if isinstance(s, str) else abs(s))
    return df

# Load the unit and unitless datasets
df_unit = clean_dataframe("/home/student/Documents/MAS500/unit.csv")
df_unitless = clean_dataframe("/home/student/Documents/MAS500/unitless.csv")

# === Plot helper ===
def plot_comparison_cross_section(df1, df2, fixed_axis, fixed_value, variable_axis, label1, label2, title_suffix):
    df1_section = df1[df1[fixed_axis] == fixed_value].copy()
    df2_section = df2[df2[fixed_axis] == fixed_value].copy()

    if df1_section.empty or df2_section.empty:
        print(f"No data found at {fixed_axis} = {fixed_value}")
        return

    x1, z1 = df1_section[variable_axis].values, df1_section["z (m)"].values
    x2, z2 = df2_section[variable_axis].values, df2_section["z (m)"].values

    # Sort for smooth line plotting
    idx1 = np.argsort(x1)
    idx2 = np.argsort(x2)

    x1, z1 = x1[idx1], z1[idx1]
    x2, z2 = x2[idx2], z2[idx2]

    # Set font size for all text
    plt.figure(figsize=(12, 7))
    plt.plot(x1, z1, label=label1, linewidth=4, color="blue")
    plt.plot(x2, z2, label=label2, linewidth=4, color="green")

    # Increase font sizes for all text elements
    plt.xlabel(variable_axis, fontsize=24)
    plt.ylabel("z (m)", fontsize=24)
    plt.title(f"Absolute Z Comparison at {fixed_axis} = {fixed_value} ({title_suffix})", fontsize=24)
    plt.legend(fontsize=24)
    plt.grid(False)
    plt.tight_layout()

    # Increase tick label sizes
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.show()

# === Visualize cross-sections comparing unit and unitless datasets ===
plot_comparison_cross_section(
    df_unit, df_unitless,
    fixed_axis="py", fixed_value=120,
    variable_axis="x (m)",
    label1="Unit", label2="Unitless",
    title_suffix="X-Slice"
)

plot_comparison_cross_section(
    df_unit, df_unitless,
    fixed_axis="px", fixed_value=160,
    variable_axis="y (m)",
    label1="Unit", label2="Unitless",
    title_suffix="Y-Slice"
)
