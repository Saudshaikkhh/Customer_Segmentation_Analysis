import pandas as pd
import joblib

def predict_new_customers(data_path):
    """Use trained model to predict clusters for new data"""
    # Load artifacts
    try:
        model = joblib.load('customer_clustering_model.pkl')
        scaler = joblib.load('customer_scaler.pkl')
        print("✅ Model and scaler loaded successfully")
    except FileNotFoundError:
        print("❌ Error: Model files not found. Train model first.")
        return
    
    # Load and prepare new data
    try:
        new_data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {data_path}")
        return
    
    # Ensure required columns exist
    required_cols = ['Income', 'MntTotal', 'In_relationship']
    missing_cols = [col for col in required_cols if col not in new_data.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return
    
    # Scale features
    X = new_data[required_cols]
    X_scaled = scaler.transform(X)
    
    # Predict clusters
    new_data['Cluster'] = model.predict(X_scaled)
    
    # Add predictions to data
    new_data.to_csv('new_customers_with_clusters.csv', index=False)
    print("💾 Saved predictions: new_customers_with_clusters.csv")
    
    # Display results - flexible column printing
    print("\n📊 Prediction results (first 5 rows):")
    
    # Determine which columns to display
    display_cols = ['Cluster'] + required_cols
    if 'ID' in new_data.columns:
        display_cols = ['ID'] + display_cols
    if 'Age' in new_data.columns:  # Example of additional optional column
        display_cols.append('Age')
    
    print(new_data[display_cols].head())
    
    return new_data

if __name__ == "__main__":
    # NEW_DATA_PATH = 'cleaned_data.csv'  # Only if it has the required columns
    NEW_DATA_PATH = 'new_customers_with_clusters.csv'  # Recommended - create this file with your new data
    predict_new_customers(NEW_DATA_PATH)