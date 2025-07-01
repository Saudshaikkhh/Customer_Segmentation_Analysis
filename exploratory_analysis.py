import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    """Perform exploratory data analysis and create visual dashboard"""
    plt.figure(figsize=(20, 16))
    plt.suptitle('iFood Customer Analysis Dashboard', fontsize=20, fontweight='bold')
    
    # Grid layout
    gs = plt.GridSpec(3, 3)
    
    # Plot 1: Spending distribution
    ax1 = plt.subplot(gs[0, 0])
    sns.boxplot(y=df['MntTotal'], ax=ax1, color='skyblue')
    ax1.set_title('Total Spending Distribution', fontsize=14)
    
    # Plot 2: Income distribution
    ax2 = plt.subplot(gs[0, 1])
    sns.histplot(df['Income'], bins=30, kde=True, ax=ax2, color='salmon')
    ax2.set_title('Income Distribution', fontsize=14)
    
    # Plot 3: Age distribution
    ax3 = plt.subplot(gs[0, 2])
    sns.histplot(df['Age'], bins=30, kde=True, ax=ax3, color='gold')
    ax3.set_title('Age Distribution', fontsize=14)
    ax3.annotate(f"Skewness: {df['Age'].skew():.2f}\nKurtosis: {df['Age'].kurt():.2f}", 
                 xy=(0.7, 0.85), xycoords='axes fraction',
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    
    # Plot 4: Marital status spending
    ax4 = plt.subplot(gs[1, 0])
    marital_avg = df.groupby('Marital')['MntTotal'].mean().sort_values()
    sns.barplot(y=marital_avg.index, x=marital_avg.values, ax=ax4, palette='viridis', orient='h')
    ax4.set_title('Average Spending by Marital Status', fontsize=14)
    
    # Plot 5: Relationship spending
    ax5 = plt.subplot(gs[1, 1])
    relationship_avg = df.groupby('In_relationship')['MntTotal'].mean()
    sns.barplot(x=relationship_avg.index, y=relationship_avg.values, ax=ax5, palette='coolwarm')
    ax5.set_xticks([0, 1])
    ax5.set_xticklabels(['Single', 'In Relationship'])
    ax5.set_title('Spending by Relationship Status', fontsize=14)
    
    # Plot 6: Correlation matrix
    ax6 = plt.subplot(gs[1, 2])
    corr_matrix = df[['MntTotal', 'Income', 'Age', 'Kidhome', 'Teenhome']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax6)
    ax6.set_title('Feature Correlation Matrix', fontsize=14)
    
    # Plot 7: Product correlations
    ax7 = plt.subplot(gs[2, :])
    product_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    sns.heatmap(df[product_cols].corr(), annot=True, cmap='RdYlGn', fmt=".2f", ax=ax7)
    ax7.set_title('Product Category Correlations', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('customer_analysis_dashboard.png', dpi=300)
    print(" Saved customer_analysis_dashboard.png")
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('cleaned_data.csv')
    perform_eda(df)
    print("\n Exploratory analysis complete!")