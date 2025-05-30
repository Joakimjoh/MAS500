import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("/home/student/Desktop/current_model_ai/training_results_v2/model_selection_metrics.csv")

# Set the model names as index (optional, for cleaner plots)
df.set_index('Model', inplace=True)

# Plot each metric
metrics = ['Box(P)', 'Recall', 'mAP50', 'mAP50-95']
for metric in metrics:
    plt.figure(figsize=(8, 5))
    df[metric].plot(kind='bar')
    plt.title(f'{metric} by Model')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()
