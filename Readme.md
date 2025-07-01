# 🛒 Customer Segmentation Analysis

A comprehensive machine learning project for customer segmentation using K-Means clustering on iFood customer data. This project provides end-to-end analysis from data preparation to model deployment, enabling businesses to understand customer behavior and create targeted marketing strategies.

## 🎯 Overview

This project performs customer segmentation analysis using machine learning clustering techniques. By analyzing customer spending patterns, income levels, and relationship status, the model identifies distinct customer segments that can be used for:

- **Targeted Marketing**: Create personalized campaigns for different customer groups
- **Product Strategy**: Develop products tailored to specific segments
- **Customer Retention**: Identify high-value customers and at-risk segments
- **Business Intelligence**: Gain insights into customer behavior patterns

## ✨ Features

- **Data Preprocessing**: Automated data cleaning and feature engineering
- **Exploratory Data Analysis**: Comprehensive visual dashboard with 7+ charts
- **Clustering Model**: K-Means clustering with optimal cluster determination
- **Model Persistence**: Save and load trained models for production use
- **Prediction Pipeline**: Classify new customers into existing segments
- **Visualization**: PCA-based cluster visualization and analysis dashboard

## 📁 Project Structure

```
customer-segmentation-analysis/
│
├── data_preparation.py          # Data loading and preprocessing
├── exploratory_analysis.py      # EDA and visualization dashboard
├── clustering_model.py          # Model training and evaluation
├── predict_clusters.py          # Prediction pipeline for new data
│
├── data/
│   ├── ifood_df.csv            # Original dataset (user provided)
│   ├── cleaned_data.csv         # Processed dataset
│   └── segmented_customers.csv  # Final results with clusters
│
├── models/
│   ├── customer_clustering_model.pkl  # Trained K-Means model
│   └── customer_scaler.pkl            # Feature scaler
│
├── outputs/
│   ├── customer_analysis_dashboard.png
│   ├── cluster_visualization.png
│   └── cluster_metrics.png
│
└── README.md
```

## 🔧 Installation

### Prerequisites

- Python 3.7+
- Required packages listed below

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/customer-segmentation-analysis.git
   cd customer-segmentation-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```

3. **Prepare your data**
   - Place your dataset as `ifood_df.csv` in the project directory
   - Update the `DATA_PATH` in `data_preparation.py` if needed

## 🚀 Usage

### Step 1: Data Preparation
```bash
python data_preparation.py
```
This script will:
- Load the raw dataset
- Remove outliers and redundant columns
- Engineer new features (marital status, relationship indicators)
- Save cleaned data as `cleaned_data.csv`

### Step 2: Exploratory Data Analysis
```bash
python exploratory_analysis.py
```
Generates a comprehensive analysis dashboard showing:
- Spending and income distributions
- Age demographics
- Marital status analysis
- Feature correlations
- Product category relationships

### Step 3: Train Clustering Model
```bash
python clustering_model.py
```
This will:
- Determine optimal number of clusters using elbow method and silhouette analysis
- Train K-Means model with 4 clusters
- Create PCA visualization
- Save model artifacts and segmented data

### Step 4: Predict New Customers
```bash
python predict_clusters.py
```
Use the trained model to classify new customers into existing segments.

## 📊 Data Requirements

Your dataset should include the following columns:

### Required Columns:
- `Income`: Customer annual income
- `MntTotal`: Total amount spent by customer
- `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`: Spending on different product categories

### Marital Status Columns (at least one):
- `marital_Divorced`, `marital_Married`, `marital_Single`, `marital_Together`, `marital_Widow`

### Optional Columns:
- `Age`: Customer age
- `Kidhome`, `Teenhome`: Number of children/teenagers
- `Z_CostContact`, `Z_Revenue`: (Will be removed during preprocessing)

## 🤖 Model Details

### Algorithm: K-Means Clustering
- **Features Used**: Income, Total Spending (MntTotal), Relationship Status
- **Preprocessing**: StandardScaler for feature normalization
- **Optimal Clusters**: 4 (determined via elbow method and silhouette analysis)
- **Random State**: 7 (for reproducibility)

### Model Selection Process:
1. **Elbow Method**: Analyze inertia vs. number of clusters
2. **Silhouette Analysis**: Evaluate cluster quality
3. **PCA Visualization**: 2D representation of clusters

### Feature Engineering:
- `In_relationship`: Binary indicator (Married or Together = 1, else = 0)
- `Marital`: Categorical variable from one-hot encoded columns
- Outlier removal using IQR method on `MntTotal`

## 📈 Results

The model identifies 4 distinct customer segments:

### Typical Cluster Profiles:
- **Cluster 0**: Budget-conscious customers (Lower income, minimal spending)
- **Cluster 1**: Premium customers (High income, high spending)
- **Cluster 2**: Average customers (Moderate income and spending)
- **Cluster 3**: Relationship-focused segment (Specific spending patterns based on relationship status)

### Model Performance:
- Clusters are well-separated in PCA space
- Silhouette score indicates good cluster quality
- Profiles show meaningful business interpretations

## 📄 File Descriptions

### Core Scripts:

**`data_preparation.py`**
- Loads raw customer data
- Handles missing values and outliers
- Engineers relationship and marital status features
- Exports cleaned dataset

**`exploratory_analysis.py`**
- Creates comprehensive 9-plot dashboard
- Analyzes spending patterns, demographics, and correlations
- Generates publication-ready visualizations

**`clustering_model.py`**
- Implements K-Means clustering pipeline
- Determines optimal cluster count
- Trains and saves model artifacts
- Creates cluster visualizations and profiles

**`predict_clusters.py`**
- Loads trained model for inference
- Classifies new customers into existing segments
- Handles data validation and error checking

### Output Files:

**`cleaned_data.csv`**: Preprocessed dataset ready for modeling
**`segmented_customers.csv`**: Original data with cluster assignments
**`customer_clustering_model.pkl`**: Trained K-Means model
**`customer_scaler.pkl`**: Fitted StandardScaler for preprocessing

## 🛠️ Customization

### Adjusting Clusters:
Modify `optimal_clusters` in `clustering_model.py`:
```python
optimal_clusters = 5  # Change from 4 to desired number
```

### Adding Features:
Update `cluster_cols` in `clustering_model.py`:
```python
cluster_cols = ['Income', 'MntTotal', 'In_relationship', 'Age']  # Add 'Age'
```

### Changing Data Path:
Update `DATA_PATH` in `data_preparation.py`:
```python
DATA_PATH = 'path/to/your/dataset.csv'
```

## 🔍 Troubleshooting

### Common Issues:

**File Not Found Error**
- Ensure dataset is in correct location
- Check file path in `data_preparation.py`

**Missing Columns Error**
- Verify your dataset has required columns
- Check column names match exactly

**Model Loading Error**
- Run clustering model training first
- Ensure model files exist in project directory

## 📊 Sample Output

```
✅ Data loaded successfully!
📊 Initial shape: 2240 rows, 29 columns
💾 Saved cleaned_data.csv

🔍 Cluster Profiles:
         Income    MntTotal  In_relationship
Cluster                                     
0        36849.0    158.3              0.3
1        68525.2   1362.8              0.7
2        51891.1    391.2              0.8
3        77236.4    771.4              0.4

💾 Saved model: customer_clustering_model.pkl
💾 Saved segmented_customers.csv
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Author**: [Shaikh Mohammed Saud]
**Email**: [shaikhmohdsaud2004@gmail.com]
**LinkedIn**: [[Your LinkedIn Profile](https://www.linkedin.com/in/saudshaikkhh/)]
**Project Link**: [https://github.com/saudshaikkhh/customer-segmentation-analysis](https://github.com/saudshaikkhh/customer-segmentation-analysis)

---

⭐ **If you found this project helpful, please give it a star!** ⭐
