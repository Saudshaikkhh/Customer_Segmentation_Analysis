import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def train_and_save_model(df):
    """Train clustering model and save artifacts"""
    # Prepare features
    cluster_cols = ['Income', 'MntTotal', 'In_relationship']
    X = df[cluster_cols]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal clusters
    inertia, silhouette = [], []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=7, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        inertia.append(kmeans.inertia_)
        silhouette.append(silhouette_score(X_scaled, clusters))
    
    # Plot metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(k_range, inertia, marker='o', color='#367ba7')
    ax1.set_title('Inertia vs. Clusters')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    
    ax2.plot(k_range, silhouette, marker='o', color='#367ba7')
    ax2.set_title('Silhouette Score vs. Clusters')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    plt.tight_layout()
    plt.savefig('cluster_metrics.png', dpi=300)
    plt.show()
    
    # Final model training
    optimal_clusters = 4  # Based on elbow method
    model = KMeans(n_clusters=optimal_clusters, random_state=7, n_init=10)
    df['Cluster'] = model.fit_predict(X_scaled)
    
    # Save model artifacts
    joblib.dump(model, 'customer_clustering_model.pkl')
    joblib.dump(scaler, 'customer_scaler.pkl')
    print("💾 Saved model: customer_clustering_model.pkl")
    print("💾 Saved scaler: customer_scaler.pkl")
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    
    # Visualize clusters
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis', s=80, alpha=0.8)
    plt.title('Customer Segmentation Clusters', fontsize=16)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.savefig('cluster_visualization.png', dpi=300)
    plt.show()
    
    # Cluster analysis
    cluster_profile = df.groupby('Cluster')[cluster_cols].mean()
    print("\n🔍 Cluster Profiles:")
    print(cluster_profile)
    
    # Save clustered data
    df.to_csv('segmented_customers.csv', index=False)
    print(" Saved segmented_customers.csv")
    
    return model, scaler

if __name__ == "__main__":
    df = pd.read_csv('cleaned_data.csv')
    model, scaler = train_and_save_model(df)
    print("\n Clustering model trained and saved!")