import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Load the dataset
file_path = 'correlation_dataset_200_rows.xlsx'
df = pd.read_excel(file_path)

# Define the pairs for analysis
pairs = [
    ('Hours_Studied', 'Test_Score'),
    ('Hours_Studied', 'Stress_Level'),
    ('Hours_Studied', 'Coffee_Cups'),
    ('Hours_Studied', 'Commute Time (minute)'),
    ('Test_Score', 'Commute Time (minute)')
]

# Create a DataFrame to store results
results = []

# Loop through each pair and compute correlations
for x, y in pairs:
    # Drop missing values for accurate calculation
    valid_df = df[[x, y]].dropna()
    x_data = valid_df[x]
    y_data = valid_df[y]
    
    # Pearson Correlation
    pearson_corr, _ = pearsonr(x_data, y_data)
    
    # Spearman Correlation
    spearman_corr, _ = spearmanr(x_data, y_data)
    
    # Append to results
    results.append({
        'Pair': f'{x} vs {y}',
        'Pearson Correlation': pearson_corr,
        'Spearman Correlation': spearman_corr
    })
    
    # Scatter Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(x_data, y_data, alpha=0.7)
    plt.title(f'Scatter Plot: {x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'scatter_{x}_vs_{y}.png')  # Save plot as image
    plt.close()

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Export results to Excel
output_file = 'Tugas Praktikum Korelasi Jason.xlsx'
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    results_df.to_excel(writer, sheet_name='Correlations', index=False)
    workbook = writer.book
    worksheet = writer.sheets['Correlations']
    
    # Insert scatter plots into Excel
    row = 2
    for i, (x, y) in enumerate(pairs):
        worksheet.insert_image(row + i * 20, 3, f'scatter_{x}_vs_{y}.png')

print(f"Results saved to '{output_file}'")