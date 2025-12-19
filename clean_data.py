import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def clean_and_transform_data(input_file, output_cleaned, output_clustering):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # --- 1. Clean Missing and Invalid Values ---
    print("Cleaning missing and invalid values...")
    
    # Replace '.' with NaN
    df.replace('.', np.nan, inplace=True)
    
    # Convert all columns to numeric where possible, coercing errors to NaN
    # This handles cases where '.' was in a numeric column
    for col in df.columns:
        # Try to convert to numeric, if it fails (e.g. real strings), keep as is for now
        # But for mixed types, we want to force numeric if it's meant to be numeric
        # We'll do a pass on specific known numeric columns first or just rely on pd.to_numeric with errors='ignore'
        # A safer approach for this dataset (which seems to have many encoded cols) is to handle specific known numeric cols
        pass 

    # Identify numeric columns (excluding the obvious string categorical ones for now)
    # We will treat negative values as NaN for typical continuous variables
    # Common AHS numeric vars: AGE1, ZINC2, COSTMED, VALUE, BEDRMS, PER, ROOMS, etc.
    numeric_cols = ['AGE1', 'ZINC2', 'COSTMED', 'VALUE', 'BEDRMS', 'PER', 'ROOMS', 'FMR', 'LMED']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] < 0, col] = np.nan

    # Drop columns with > 30% missing values
    missing_threshold = 0.3
    missing_counts = df.isnull().mean()
    cols_to_drop = missing_counts[missing_counts > missing_threshold].index.tolist()
    print(f"Dropping {len(cols_to_drop)} columns with > 30% missing values: {cols_to_drop[:5]}...")
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Impute remaining missing values
    # Numeric -> Median, Categorical (Object) -> Mode
    for col in df.columns:
        if df[col].dtype == np.number:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            # For object columns, fill with mode
            if not df[col].mode().empty:
                df[col].fillna(df[col].mode()[0], inplace=True)

    # --- 2. Normalize Categorical Affordability and AMI Columns ---
    print("Normalizing categorical columns...")
    
    # Function to extract first number from string
    def extract_code(val):
        if isinstance(val, str):
            # Split by space and take the first part
            parts = val.split(' ')
            if parts:
                try:
                    # Handle cases like '1' or '1.0' or '1.'
                    return float(parts[0])
                except ValueError:
                    return np.nan # Or keep original if it's not a number code
        return val

    # Identify columns that look like "Code Label" (often start with FMT)
    # We'll target specific ones mentioned or common in AHS
    fmt_cols = [c for c in df.columns if c.startswith('FMT')]
    
    for col in fmt_cols:
        # Apply extraction
        # We need to ensure we don't lose data if it's already numeric
        if df[col].dtype == object:
             # Extract the leading number
             df[col] = df[col].astype(str).apply(extract_code)
             # Convert to numeric
             df[col] = pd.to_numeric(df[col], errors='coerce')
             
             # Fill any NaNs created by this process (if any) with 0 or mode, 
             # but usually these fields are well-formed. Let's fill with mode to be safe.
             if df[col].isnull().any():
                 df[col].fillna(df[col].mode()[0], inplace=True)

    # Ensure REGION and METRO3 are numeric (they usually are, but good to check)
    if 'REGION' in df.columns:
        df['REGION'] = pd.to_numeric(df['REGION'], errors='coerce').fillna(0)
    if 'METRO3' in df.columns:
        df['METRO3'] = pd.to_numeric(df['METRO3'], errors='coerce').fillna(0)

    # --- 3. Create Derived Affordability Metrics ---
    print("Creating derived metrics...")
    
    # cost_burden_ratio = (Housing Cost * 12) / Household Income
    # Use COSTMED (Monthly housing cost) and ZINC2 (Household income)
    if 'COSTMED' in df.columns and 'ZINC2' in df.columns:
        # Avoid division by zero
        income = df['ZINC2'].replace(0, 1) 
        df['cost_burden_ratio'] = (df['COSTMED'] * 12) / income
        
        # Cap ratio at reasonable value (e.g., 2.0 or 3.0) to avoid outliers skewing clustering
        df['cost_burden_ratio'] = df['cost_burden_ratio'].clip(upper=3.0)
    else:
        print("Warning: COSTMED or ZINC2 missing, cannot calculate cost_burden_ratio")
        df['cost_burden_ratio'] = 0

    # affordability_status
    # < 30% -> 1 (Affordable)
    # 30-50% -> 2 (Moderately Burdened)
    # > 50% -> 3 (Severely Burdened)
    conditions = [
        (df['cost_burden_ratio'] < 0.3),
        (df['cost_burden_ratio'] >= 0.3) & (df['cost_burden_ratio'] <= 0.5),
        (df['cost_burden_ratio'] > 0.5)
    ]
    choices = [1, 2, 3]
    df['affordability_status'] = np.select(conditions, choices, default=1)

    # --- 4. Feature Reduction and Optimization ---
    print("Selecting features...")
    
    # Candidate features for clustering
    # We prefer the FMT versions if available as they are often categorical codes
    # But we cleaned them to be numeric codes now.
    
    potential_features = [
        'ZINC2', 'COSTMED', 'AGE1', 'BEDRMS', 'PER', 'ROOMS', 'VALUE',
        'FMTSTRUCTURETYPE', 'FMTOWNRENT', 'FMTSTATUS', 'REGION', 'METRO3',
        'FMTINCRELAMICAT', 'cost_burden_ratio', 'affordability_status'
    ]
    
    # Filter for columns that actually exist
    selected_features = [c for c in potential_features if c in df.columns]
    
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    clustering_df = df[selected_features].copy()

    # --- 5. Standardize and Scale ---
    print("Scaling features...")
    scaler = MinMaxScaler()
    clustering_df_scaled = pd.DataFrame(scaler.fit_transform(clustering_df), columns=clustering_df.columns)

    # --- 6. Add or Enhance Geographic Identifiers ---
    print("Generating synthetic geographic labels...")
    
    # Synthetic locality label: REGION * 10 + METRO3
    # This creates a unique code for Region-Metro combinations
    if 'REGION' in df.columns and 'METRO3' in df.columns:
        df['Locality_Label'] = df['REGION'] * 10 + df['METRO3']
        clustering_df_scaled['Locality_Label'] = df['Locality_Label'] # Add to clustering features too? 
        # Usually we might not cluster ON the label itself if it's nominal, 
        # but the user asked to "Add or Enhance Geographic Identifiers... to support segmentation"
        # We'll add it to the cleaned data. For clustering, REGION and METRO3 are already there.
    
    # --- 7. Output Requirements ---
    print("Saving outputs...")
    
    # (a) cleaned_data.csv (all cleaned and transformed fields)
    df.to_csv(output_cleaned, index=False)
    print(f"Saved {output_cleaned}")
    
    # (b) clustering_features.csv (only final encoded & scaled features)
    clustering_df_scaled.to_csv(output_clustering, index=False)
    print(f"Saved {output_clustering}")
    
    print("Done.")

if __name__ == "__main__":
    input_csv = "cleaned_full_affordability.csv"
    output_cleaned_csv = "cleaned_data.csv"
    output_clustering_csv = "clustering_features.csv"
    
    clean_and_transform_data(input_csv, output_cleaned_csv, output_clustering_csv)
