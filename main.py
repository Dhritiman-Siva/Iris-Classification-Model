import pandas as pd

# 1. Load the dataset
data = pd.read_excel('IRIS.xlsx')

print("--- Initial Data Overview ---")
print(f"Initial shape: {data.shape}")
print(data.info())

# 2. Check for missing / null values
print("\n--- Missing Values Check ---")
missing_vals = data.isnull().sum()
print(missing_vals)

# If any missing values existed, we would drop or impute them:
# data = data.dropna()

# 3. Check and remove duplicate rows
duplicates_count = data.duplicated().sum()
print(f"\n--- Duplicate Rows ---")
print(f"Number of duplicate rows found: {duplicates_count}")

if duplicates_count > 0:
    data = data.drop_duplicates().reset_index(drop=True)
    print(f"Duplicates removed. New shape: {data.shape}")

# 4. Standardize / Clean categorical values
# Clean species column (strip spaces, optional format standardizing)
data['species'] = data['species'].astype(str).str.strip()

# 5. Verify numeric column types and check summary statistics
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for col in feature_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

print("\n--- Cleaned Data Summary ---")
print(data.describe())

print("\n--- Species Distribution (Cleaned) ---")
print(data['species'].value_counts())

# 6. Save the cleaned dataset for further analysis
output_file = 'iris_cleaned.csv'
data.to_csv(output_file, index=False)
print(f"\nCleaned dataset saved successfully to '{output_file}'.")
