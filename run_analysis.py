import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_analysis(features_file, metadata_file, output_dir="."):
    print("--- Phase 1 & 2: Load & Validate Data ---")
    print(f"Loading features from {features_file}...")
    X = pd.read_csv(features_file)
    print(f"Loading metadata from {metadata_file}...")
    metadata = pd.read_csv(metadata_file)
    
    # Ensure alignment
    if len(X) != len(metadata):
        print("Error: Features and Metadata length mismatch!")
        return

    print(f"Data loaded. Shape: {X.shape}")

    print("\n--- Phase 3: Clustering Model Development ---")
    
    # 1. Optimal K Search (Elbow Method)
    print("Determining optimal k...")
    inertia = []
    sil_scores = []
    K_range = range(2, 11)
    
    # Sample for silhouette score to save time if dataset is huge
    sample_size = min(10000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        
        # Silhouette on sample
        labels = kmeans.predict(X_sample)
        score = silhouette_score(X_sample, labels)
        sil_scores.append(score)
        print(f"  k={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={score:.3f}")

    # Plot Elbow
    plt.figure(figsize=(10, 5))
    plt.plot(K_range, inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig(f"{output_dir}/elbow_plot.png")
    plt.close()
    print("Saved elbow_plot.png")

    # Select optimal k automatically (simplified logic: max silhouette or fixed heuristic)
    # We'll pick the k with the highest silhouette score
    optimal_k = K_range[np.argmax(sil_scores)]
    print(f"Optimal k selected based on Silhouette Score: {optimal_k}")

    # 2. Train Final KMeans Model
    print(f"Training KMeans with k={optimal_k}...")
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X)
    
    # Save the model
    joblib.dump(kmeans_final, f"{output_dir}/kmeans_model.pkl")
    print("Saved kmeans_model.pkl")
    
    # 3. Train DBSCAN (Comparison)
    print("Training DBSCAN (eps=0.3, min_samples=20)...")
    # Using conservative params for high-dim scaled data
    dbscan = DBSCAN(eps=0.3, min_samples=20)
    dbscan_labels = dbscan.fit_predict(X)
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"DBSCAN found {n_dbscan_clusters} clusters (excluding noise).")

    print("\n--- Phase 4: Cluster Interpretation & Profiling ---")
    
    # Add labels to metadata
    metadata['Cluster_Label'] = cluster_labels
    
    # PCA for Visualization
    print("Performing PCA for visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='viridis', s=10)
    plt.title(f'KMeans Clusters (k={optimal_k}) - PCA Projection')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Cluster')
    plt.savefig(f"{output_dir}/pca_clusters.png")
    plt.close()
    print("Saved pca_clusters.png")

    # Profiling
    print("Generating cluster profiles...")
    # Select key columns for summary (mix of original values)
    # We need to make sure we use the columns that exist in refined_metadata
    summary_cols = ['ZINC2', 'COSTMED', 'cost_burden_ratio', 'AGE1', 'BEDRMS', 'VALUE', 'FMTOWNRENT']
    # Check availability
    available_cols = [c for c in summary_cols if c in metadata.columns]
    
    profile = metadata.groupby('Cluster_Label')[available_cols].mean().round(2)
    profile['Count'] = metadata['Cluster_Label'].value_counts()
    profile['Percent'] = (profile['Count'] / len(metadata) * 100).round(1)
    
    print("\nCluster Summary Table:")
    print(profile)
    profile.to_csv(f"{output_dir}/cluster_summary.csv")
    
    # Save assignments
    metadata[['Cluster_Label']].to_csv(f"{output_dir}/cluster_assignments.csv", index=True)

    print("\n--- Phase 5: Smart City Policy Mapping ---")
    
    # Generate automated insights based on profile
    recommendations = []
    for cluster_id, row in profile.iterrows():
        rec = f"Cluster {cluster_id}: "
        
        # Income Logic
        if row['ZINC2'] < 30000:
            rec += "Low Income. "
        elif row['ZINC2'] > 80000:
            rec += "High Income. "
        else:
            rec += "Middle Income. "
            
        # Burden Logic
        if row['cost_burden_ratio'] > 0.5:
            rec += "Severely Burdened. POLICY: Rent subsidies, emergency assistance."
        elif row['cost_burden_ratio'] > 0.3:
            rec += "Moderately Burdened. POLICY: Affordable housing development, zoning incentives."
        else:
            rec += "Affordable. POLICY: Market monitoring, maintain stability."
            
        recommendations.append(rec)
    
    print("\nAutomated Policy Recommendations:")
    for r in recommendations:
        print(f"- {r}")
        
    # Save recommendations to text file
    with open(f"{output_dir}/policy_recommendations.txt", "w") as f:
        f.write("\n".join(recommendations))

    print("\n--- Phase 6: Final Output Generation ---")
    # Cost Burden Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster_Label', y='cost_burden_ratio', data=metadata)
    plt.title('Cost Burden Ratio by Cluster')
    plt.ylim(0, 1.5) # Cap y-axis for readability
    plt.savefig(f"{output_dir}/burden_distribution.png")
    plt.close()
    print("Saved burden_distribution.png")
    
    print("Analysis Complete.")

if __name__ == "__main__":
    run_analysis("refined_features.csv", "refined_metadata.csv")
