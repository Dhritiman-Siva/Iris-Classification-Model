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

# 7. 80/20 Random Train-Test Split
from sklearn.model_selection import train_test_split

# Separate features (X) and target label (y)
X = data[feature_cols]
y = data['species']

# Split dataset: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--- 80/20 Dataset Split ---")
print(f"Total samples: {len(data)}")
print(f"Training set (80%): {X_train.shape[0]} samples (Features: {X_train.shape})")
print(f"Testing set (20%): {X_test.shape[0]} samples (Features: {X_test.shape})")

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())
