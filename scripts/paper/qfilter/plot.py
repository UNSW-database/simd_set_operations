# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

result_dir = "../../processed/qfilter-broadwell"

scalar_path = os.path.join(result_dir, 'ScalarMerge.csv')
scalar_df = pd.read_csv(scalar_path)
scalar_runtime = scalar_df['Runtime_us']
print(scalar_runtime)

# Initialize the plot
plt.figure(figsize=(10, 6))

# Loop through each file in the directory
for file in os.listdir(result_dir):
    if file.endswith(".csv"):
        # Construct the full file path
        file_path = os.path.join(result_dir, file)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        print(df['Throughput_epus'])
        # Calculate throughput relative to Shuffling
        relative_speed = scalar_runtime / df['Runtime_us']
        # Plot selectivity vs. relative throughput, using the file name as the label
        plt.plot(df['Selectivity'], relative_speed, label=file.replace('.csv', ''))

# Customize the plot
plt.title('Relative Speed vs. Selectivity (Original QFilter experiment)')
plt.xlabel('Selectivity')
plt.ylabel('Relative Speed (Scalar=1)')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# %%
