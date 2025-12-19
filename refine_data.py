import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

def refine_data(input_file, output_features, output_metadata):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    # --- PHASE A: Remove Non-Useful or Harmful Columns ---
    print("Phase A: Removing non-useful columns and cleaning negatives...")
    
    # 1. Remove identifiers
    if 'CONTROL' in df.columns:
        df.drop(columns=['CONTROL'], inplace=True)
        
    # 2. Global clean of negative placeholders
    # Common AHS codes: -6 (Not applicable), -9 (Not reported), -5, -7, -8
    # We treat ALL negative values in numeric columns as NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Count negatives before
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            # print(f"  {col}: replacing {neg_count} negative values with NaN")
            df.loc[df[col] < 0, col] = np.nan

    # --- PHASE B: Feature Selection & Dimensionality Reduction ---
    print("Phase B: Feature Selection & Correlation Filtering...")
    
    # 4. Select relevant variables
    # We prioritize the "FMT" versions if they are numeric codes (which we cleaned in step 1)
    # or the raw continuous variables.
    
    candidates = [
        'ZINC2', 'COSTMED', 'VALUE', 'FMR', # Cost/Income
        'cost_burden_ratio', 'affordability_status', # Derived
        'AGE1', 'PER', 'BEDRMS', 'ROOMS', # Demographics
        'FMTSTRUCTURETYPE', 'FMTOWNRENT', 'FMTSTATUS', # Housing Type
        'REGION', 'METRO3', 'Locality_Label', # Geo
        'FMTINCRELAMICAT' # AMI
    ]
    
    # Filter for what exists
    selected_features = [c for c in candidates if c in df.columns]
    
    # Create a subset for analysis
    feature_df = df[selected_features].copy()
    
    # 5. Correlation Matrix & Removal
    # Drop columns with > 30% missing (again, just to be safe after negative cleaning)
    missing_ratio = feature_df.isnull().mean()
    drop_missing = missing_ratio[missing_ratio > 0.3].index.tolist()
    if drop_missing:
        print(f"  Dropping high-missing cols: {drop_missing}")
        feature_df.drop(columns=drop_missing, inplace=True)
    
    # Impute BEFORE correlation to avoid errors
    print("  Imputing missing values for correlation check...")
    for col in feature_df.columns:
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    # Compute correlation
    corr_matrix = feature_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation > 0.85
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    
    print(f"  High correlation features to drop (>0.85): {to_drop}")
    
    # We might want to be selective. 
    # e.g. if 'affordability_status' is correlated with 'cost_burden_ratio', 
    # we usually prefer 'cost_burden_ratio' (continuous) for clustering.
    # The automatic drop usually drops the *second* one encountered.
    # Let's refine the drop list manually if needed, but automatic is okay for this task.
    
    final_features = feature_df.drop(columns=to_drop)
    print(f"  Final Feature Count: {final_features.shape[1]}")
    print(f"  Final Features: {final_features.columns.tolist()}")

    # --- PHASE C: Handle Missing or Invalid Values ---
    # (Already imputed above for correlation, but let's ensure the final set is clean)
    # We use the imputed values from above
    
    # --- PHASE D: Scaling & Normalization ---
    print("Phase D: Scaling...")
    scaler = MinMaxScaler()
    scaled_matrix = scaler.fit_transform(final_features)
    scaled_df = pd.DataFrame(scaled_matrix, columns=final_features.columns)
    
    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("Saved scaler.pkl")

    # --- PHASE E: Export Final Datasets ---
    print("Phase E: Exporting...")
    
    # refined_features.csv (The matrix for clustering)
    scaled_df.to_csv(output_features, index=False)
    
    # refined_metadata.csv (Original columns + cluster-ready fields, for interpretation)
    # We take the original df, add the cleaned features (unscaled) to it or just save the cleaned df
    # Let's save the cleaned version of the selected features + original context
    # We'll just save the full cleaned df again but with the rigorous negative cleaning applied
    df.to_csv(output_metadata, index=False)
    
    print(f"Saved {output_features}")
    print(f"Saved {output_metadata}")

if __name__ == "__main__":
    refine_data("cleaned_data.csv", "refined_features.csv", "refined_metadata.csv")
