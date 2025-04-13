import pandas as pd
import numpy as np
import h2o
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
FILE_FORMAT = "jpg"  # Choose "pdf" or "jpg"
OUTPUT_DIR = "./plots"  # Directory to save the plots

import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load the dataset
data = pd.read_csv("reduced_data.csv")

def aggregate_play(df):
    """
    Aggregates the data for each play, pivoting the table so that each row represents a single play
    and contains the x, y, o, and position data for each player.
    """
    # Select the columns we need
    df = df[['uniquePlayId', 'playDirection', 'x', 'y', 'o', 'position', 'offenseFormation']]

    # Pivot the table using a multi-index
    df_pivot = df.pivot_table(index=['uniquePlayId', 'playDirection', 'offenseFormation'], columns='position', values=['x', 'y', 'o'])

    # Flatten the multi-level column index
    df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]

    # Reset index to make 'uniquePlayId' a regular column
    df_pivot = df_pivot.reset_index()

    return df_pivot

data = aggregate_play(data)

# Configure H2O
h2o_config = {
    "nthreads": -1,  # Use all available CPU threads
    "max_mem_size": "16G",  # Adjust based on your RAM
}

# Initialize H2O
h2o.init(**h2o_config)

# Load the previously saved model
model_path = "C:/Users/quort/OneDrive/Desktop/coding/model/StackedEnsemble_BestOfFamily_4_AutoML_1_20250321_14016"  # YOUR MODEL PATH
best_model = h2o.load_model(model_path)

# Convert pandas DataFrame to H2O Frame
h2o_df = h2o.H2OFrame(data)

# Identify target and predictor columns (same as before)
y = "offenseFormation"
x = h2o_df.columns
x.remove(y)
x.remove("uniquePlayId")
x.remove("playDirection")

# Split data into training and testing sets (for evaluation/plots)
train, test = h2o_df.split_frame(ratios=[0.8], seed=1234)

# Evaluate the Leader Model on Test Data
perf = best_model.model_performance(test_data=test)
print(perf)

# Access Base Models and their variable importances
base_models = best_model.base_models
variable_importances = {}

for model_id in base_models:
    try:
        model = h2o.get_model(model_id)  # Load the base model
        if hasattr(model, 'varimp'):
            varimp = model.varimp(use_pandas=True)
            if varimp is not None:
                variable_importances[model_id] = varimp
            else:
                print(f"Model {model_id} has varimp attribute but it is None.")
        else:
            print(f"Model {model_id} does not have varimp attribute.")
    except Exception as e:
        print(f"Error getting variable importance for model {model_id}: {e}")

# Aggregate Variable Importances (Example: Simple Averaging)
aggregated_importances = {}
for model_id, varimp_df in variable_importances.items():
    for index, row in varimp_df.iterrows():
        variable = row['variable']
        importance = row['relative_importance']
        if variable in aggregated_importances:
            aggregated_importances[variable].append(importance)
        else:
            aggregated_importances[variable] = [importance]

# Average the importances across models
averaged_importances = {
    variable: np.mean(importances) for variable, importances in aggregated_importances.items()
}

# Convert to DataFrame for plotting
importance_df = pd.DataFrame(
    list(averaged_importances.items()), columns=['variable', 'relative_importance']
).sort_values(by='relative_importance', ascending=False)

# **Variable Importance Plot**
plt.figure(figsize=(12, 6))
sns.barplot(x="variable", y="relative_importance", data=importance_df.head(20), palette="viridis")
plt.xticks(rotation=45, ha="right")
plt.title("Aggregated Variable Importance from Stacked Ensemble")
plt.tight_layout()

# Save the plot
plot_filename = os.path.join(OUTPUT_DIR, f"variable_importance.{FILE_FORMAT}")
plt.savefig(plot_filename, bbox_inches="tight")
print(f"Variable importance plot saved to: {plot_filename}")
plt.close()  # Close the plot to prevent it from displaying

# Since partial_plot relies on direct variable importance, it may not be directly applicable.
# Alternative: You could try partial plots on individual base models.

# Shutdown H2O cluster
leaderboard = best_model.leaderboard

# Convert to pandas DataFrame and select top 5 models
top_models = leaderboard.as_data_frame().head(5)

# Create a bar plot for top models
plt.figure(figsize=(12, 6))
sns.barplot(x='model_id', y='auc', data=top_models, palette='viridis')
plt.title('Top 5 Models in Stacked Ensemble')
plt.xlabel('Model ID')
plt.ylabel('AUC')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
plot_filename = os.path.join(OUTPUT_DIR, f"top_models.{FILE_FORMAT}")
plt.savefig(plot_filename, bbox_inches="tight")
print(f"Top models plot saved to: {plot_filename}")
plt.close()

# Get details of top models
model_details = []
for model_id in top_models['model_id']:
    model = h2o.get_model(model_id)
    details = {
        'model_id': model_id,
        'algorithm': model.algo,
        'parameters': model.params
    }
    model_details.append(details)

# Create a text file with model architectures
architecture_filename = os.path.join(OUTPUT_DIR, "model_architectures.txt")
with open(architecture_filename, 'w') as f:
    for details in model_details:
        f.write(f"Model ID: {details['model_id']}\n")
        f.write(f"Algorithm: {details['algorithm']}\n")
        f.write("Key Parameters:\n")
        for param, value in details['parameters'].items():
            if param in ['ntrees', 'max_depth', 'learn_rate', 'min_rows', 'sample_rate', 'col_sample_rate']:
                f.write(f"  {param}: {value['actual']}\n")
        f.write("\n" + "-"*50 + "\n\n")

print(f"Model architectures saved to: {architecture_filename}")

h2o.cluster().shutdown()
