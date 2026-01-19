
import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['ISIC Archive\n(2018/19/HAM10000)', 'SD-198', 'Derm7pt', 'Private/Hospital', 'PH2', 'Dermnet', 'Monkeypox']
counts = [13, 8, 6, 2, 1, 1, 1]

# Sort data for better visualization
sorted_indices = np.argsort(counts)[::-1]
datasets = [datasets[i] for i in sorted_indices]
counts = [counts[i] for i in sorted_indices]

# Aesthetic Setup
plt.figure(figsize=(10, 6))
# Use a modern color palette
colors = ['#2E86C1', '#17A589', '#D4AC0D', '#CB4335', '#884EA0', '#BA4A00', '#283747']

# Create Horizontal Bar Chart
bars = plt.barh(datasets, counts, color=colors[:len(datasets)], height=0.7)
plt.xlabel('Number of Studies', fontsize=12, fontweight='bold')
plt.title('Distribution of Dataset Usage in Primary Studies (2020-2025)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()  # Highest count at top

# Remove spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_color('#DDDDDD')

# Add grid lines
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
             f'{int(width)}', 
             ha='left', va='center', fontsize=10, fontweight='bold', color='#333333')

plt.tight_layout()

# Save
plt.savefig('images/dataset_usage_chart.png', dpi=300, bbox_inches='tight')
print("Chart generated successfully at images/dataset_usage_chart.png")
