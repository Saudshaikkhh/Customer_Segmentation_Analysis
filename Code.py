import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Ignore warnings
warnings.filterwarnings("ignore")

# ========= Load and Inspect =========
DATA_PATH = r'C:\Users\Shaikh Mohammed Saud\OneDrive\Desktop\Internship\ifood_df.csv'

try:
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully!")
except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}")
    exit(1)

# Initial data inspection
print("\nInitial Data Summary:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nMissing Values:")
print(df.isna().sum().sort_values(ascending=False))
print("\nColumn Dtypes:")
print(df.dtypes.value_counts())

# ========= Clean Data =========
df.drop(columns=['Z_CostContact', 'Z_Revenue'], inplace=True, errors='ignore')

# ========= Outlier Removal: MntTotal =========
Q1 = df['MntTotal'].quantile(0.25)
Q3 = df['MntTotal'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

original_count = df.shape[0]
df = df[(df['MntTotal'] >= lower_bound) & (df['MntTotal'] <= upper_bound)]
removed_count = original_count - df.shape[0]
print(f"\nRemoved {removed_count} outliers from MntTotal ({removed_count/original_count:.2%} of data)")

# ========= Feature Engineering =========
def get_marital_status(row):
    if row['marital_Divorced'] == 1:
        return 'Divorced'
    elif row['marital_Married'] == 1:
        return 'Married'
    elif row['marital_Single'] == 1:
        return 'Single'
    elif row['marital_Together'] == 1:
        return 'Together'
    elif row['marital_Widow'] == 1:
        return 'Widow'
    return 'Unknown'

df['Marital'] = df.apply(get_marital_status, axis=1)
df['In_relationship'] = df.apply(
    lambda row: 1 if row['marital_Married'] == 1 or row['marital_Together'] == 1 else 0, 
    axis=1
)

# ========= Create Comprehensive Grid Layout =========
plt.figure(figsize=(20, 16))
plt.suptitle('iFood Customer Analysis Dashboard', fontsize=20, fontweight='bold')

# Grid layout: 3 rows, 3 columns
gs = plt.GridSpec(3, 3, figure=plt.gcf())

# 1. MntTotal Distribution (Boxplot)
ax1 = plt.subplot(gs[0, 0])
sns.boxplot(y=df['MntTotal'], ax=ax1, color='skyblue')
ax1.set_title('Total Spending Distribution', fontsize=14)
ax1.set_ylabel('Total Spending')

# 2. Income Distribution (Boxplot)
ax2 = plt.subplot(gs[0, 1])
sns.boxplot(y=df['Income'], ax=ax2, color='lightgreen')
ax2.set_title('Income Distribution', fontsize=14)
ax2.set_ylabel('Income')

# 3. Income Histogram
ax3 = plt.subplot(gs[0, 2])
sns.histplot(df['Income'], bins=30, kde=True, ax=ax3, color='salmon')
ax3.set_title('Income Distribution', fontsize=14)
ax3.set_xlabel('Income')
ax3.set_ylabel('Frequency')

# 4. Age Distribution
ax4 = plt.subplot(gs[1, 0])
sns.histplot(df['Age'], bins=30, kde=True, ax=ax4, color='gold')
ax4.set_title('Age Distribution', fontsize=14)
ax4.set_xlabel('Age')
ax4.set_ylabel('Frequency')
ax4.annotate(f"Skewness: {df['Age'].skew():.2f}\nKurtosis: {df['Age'].kurt():.2f}", 
             xy=(0.7, 0.85), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# 5. Marital Status Spending
ax5 = plt.subplot(gs[1, 1])
marital_avg = df.groupby('Marital')['MntTotal'].mean().sort_values(ascending=False)
sns.barplot(x=marital_avg.index, y=marital_avg.values, ax=ax5, palette='viridis')
ax5.set_title('Average Spending by Marital Status', fontsize=14)
ax5.set_xlabel('Marital Status')
ax5.set_ylabel('Average Spending')
ax5.tick_params(axis='x', rotation=45)
for i, v in enumerate(marital_avg.values):
    ax5.text(i, v + 20, f"${v:.0f}", ha='center', fontsize=10)

# 6. Relationship Status Spending
ax6 = plt.subplot(gs[1, 2])
relationship_avg = df.groupby('In_relationship')['MntTotal'].mean()
sns.barplot(x=relationship_avg.index.map({0: 'Single', 1: 'In Relationship'}), 
            y=relationship_avg.values, ax=ax6, palette='coolwarm')
ax6.set_title('Spending by Relationship Status', fontsize=14)
ax6.set_xlabel('Relationship Status')
ax6.set_ylabel('Average Spending')
for i, v in enumerate(relationship_avg.values):
    ax6.text(i, v + 20, f"${v:.0f}", ha='center', fontsize=12)

# 7. Correlation Heatmap
ax7 = plt.subplot(gs[2, :])  # Span all columns in last row
cols_demographics = ['Income', 'Age']
cols_children = ['Kidhome', 'Teenhome']
corr_matrix = df[['MntTotal'] + cols_demographics + cols_children].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
            vmin=-1, vmax=1, ax=ax7, cbar_kws={'label': 'Correlation Coefficient'})
ax7.set_title('Feature Correlation Matrix', fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
plt.savefig('customer_segmentation_dashboard.png', dpi=300, bbox_inches='tight')
print("\nDashboard saved as 'customer_segmentation_dashboard.png'")
plt.show()

# ========= Additional Analysis =========
# Categorical correlations
cols_marital = ['marital_Divorced', 'marital_Married', 'marital_Single', 'marital_Together', 'marital_Widow']
cols_education = ['education_2n Cycle', 'education_Basic', 'education_Graduation', 'education_Master', 'education_PhD']

print("\n--- Marital Status Correlation with Spending ---")
marital_corrs = {}
for col in cols_marital:
    if col in df and df[col].nunique() == 2:
        corr = df[col].corr(df['MntTotal'])
        marital_corrs[col.split('_')[1]] = corr
        print(f'{col.split("_")[1]:<10}: {corr:.4f}')

print("\n--- Education Correlation with Spending ---")
education_corrs = {}
for col in cols_education:
    if col in df and df[col].nunique() == 2:
        corr = df[col].corr(df['MntTotal'])
        education_corrs[col.split('_')[1]] = corr
        print(f'{col.split("_")[1]:<10}: {corr:.4f}')

# ========= Relationship Status Insights =========
relationship_counts = df['In_relationship'].value_counts(normalize=True) * 100
print("\n--- Relationship Status Distribution ---")
print(f"Single: {relationship_counts.get(0, 0):.1f}%")
print(f"In Relationship: {relationship_counts.get(1, 0):.1f}%")

# ========= Save Processed Data =========
df.to_csv('processed_customer_data.csv', index=False)
print("\nProcessed data saved as 'processed_customer_data.csv'")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols_for_clustering = ['Income', 'MntTotal', 'In_relationship']
data_scaled = df.copy()
data_scaled[cols_for_clustering] = scaler.fit_transform(df[cols_for_clustering])
print(data_scaled[cols_for_clustering].describe())

