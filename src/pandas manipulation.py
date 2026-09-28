import os
import pandas as pd

# ---------------------------------------------------------
# 1. Load Real Indian Dataset & Inspect
# ---------------------------------------------------------
# Sample Indian State Demographic/Revenue dataset
data = {
    'State': ['Karnataka', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Maharashtra', 'Delhi'],
    'District_Code': [101, 201, 102, 301, 202, 401],
    'Population_Lakhs': [675, 1231, 675, 721, 1231, 167],
    'Revenue_Cr': [1200.5, 2500.0, 850.0, 1100.2, 1900.8, 950.0],
    'Zone': ['South', 'West', 'South', 'South', 'West', 'North']
}

df = pd.DataFrame(data)

# Print initial inspection deliverables
print("=== SHAPE ===")
print(df.shape)

print("\n=== DTYPES ===")
print(df.dtypes)

print("\n=== HEAD (10) ===")
print(df.head(10))

# ---------------------------------------------------------
# 2. Required Data Manipulation Operations
# ---------------------------------------------------------
# FILTER: Extract high revenue records (> 1000 Cr)
filtered_df = df[df['Revenue_Cr'] > 1000]
print("\n=== FILTERED DATA (Revenue > 1000 Cr) ===")
print(filtered_df)

# GROUPBY: Calculate average revenue and total population per Zone
grouped_df = df.groupby('Zone').agg({
    'Revenue_Cr': 'mean',
    'Population_Lakhs': 'sum'
}).reset_index()
print("\n=== GROUPBY AGGREGATION ===")
print(grouped_df)

# MERGE: Combine state data with a capital city lookup table
capitals = {
    'State': ['Karnataka', 'Maharashtra', 'Tamil Nadu', 'Delhi'],
    'Capital': ['Bengaluru', 'Mumbai', 'Chennai', 'New Delhi']
}
df_capitals = pd.DataFrame(capitals)
merged_df = pd.merge(df, df_capitals, on='State', how='left')
print("\n=== MERGED DATA ===")
print(merged_df)

# PIVOT_TABLE: Summarize total revenue grouped by Zone and State
pivot_df = pd.pivot_table(
    merged_df,
    values='Revenue_Cr',
    index='Zone',
    columns='State',
    aggfunc='sum',
    fill_value=0
)
print("\n=== PIVOT TABLE ===")
print(pivot_df)

# ---------------------------------------------------------
# 3. Export Files and Compare Sizes
# ---------------------------------------------------------
csv_path = 'cleaned_indian_data.csv'
parquet_path = 'cleaned_indian_data.parquet'

merged_df.to_csv(csv_path, index=False)
merged_df.to_parquet(parquet_path, index=False)

csv_size = os.path.getsize(csv_path)
parquet_size = os.path.getsize(parquet_path)

print(f"\n=== FILE SIZE COMPARISON ===")
print(f"CSV Size: {csv_size} bytes")
print(f"Parquet Size: {parquet_size} bytes")