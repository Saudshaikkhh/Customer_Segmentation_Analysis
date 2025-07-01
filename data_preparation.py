import pandas as pd
import numpy as np
import warnings

def load_and_prepare_data(file_path):
    """
    Load and prepare the dataset
    Returns cleaned DataFrame
    """
    warnings.filterwarnings("ignore")
    
    try:
        df = pd.read_csv(file_path)
        print("✅ Data loaded successfully!")
        print(f"📊 Initial shape: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        exit(1)
    
    # Remove redundant columns
    df.drop(columns=['Z_CostContact', 'Z_Revenue'], inplace=True, errors='ignore')
    
    # Outlier removal
    Q1 = df['MntTotal'].quantile(0.25)
    Q3 = df['MntTotal'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['MntTotal'] >= lower_bound) & (df['MntTotal'] <= upper_bound)]
    
    # Feature engineering
    marital_map = {
        'marital_Divorced': 'Divorced',
        'marital_Married': 'Married',
        'marital_Single': 'Single',
        'marital_Together': 'Together',
        'marital_Widow': 'Widow'
    }
    
    df['Marital'] = df[list(marital_map.keys())].idxmax(axis=1).map(marital_map)
    df['In_relationship'] = df[['marital_Married', 'marital_Together']].max(axis=1)
    
    # Save cleaned data
    df.to_csv('cleaned_data.csv', index=False)
    print("💾 Saved cleaned_data.csv")
    
    return df

if __name__ == "__main__":
    DATA_PATH = r'C:\Users\Shaikh Mohammed Saud\OneDrive\Desktop\Internship\Customer_Segmentation_Analysis\ifood_df.csv'
    df = load_and_prepare_data(DATA_PATH)
    print("\n Data preparation complete!")