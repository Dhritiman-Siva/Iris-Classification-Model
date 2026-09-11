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

# 7. Exploratory Data Analysis (EDA) & Parameter Relationships
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Set2")

print("\n" + "="*55)
print("       EXPLORATORY DATA ANALYSIS (EDA) & RELATIONSHIPS")
print("="*55)

# A. Mean and Standard Deviation per Species
print("\n--- Feature Mean per Species ---")
species_mean = data.groupby('species')[feature_cols].mean()
print(species_mean.round(2))

# B. Correlation Matrix among Parameters
corr_matrix = data[feature_cols].corr()
print("\n--- Pearson Correlation Matrix ---")
print(corr_matrix.round(3))

# Visual Plot 1: Correlation Matrix Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, vmin=-1, vmax=1)
plt.title("Correlation Matrix of Iris Flower Features", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("eda_correlation_matrix.png", dpi=300)
plt.close()
print("\n1. Correlation heatmap saved as 'eda_correlation_matrix.png'.")

# Visual Plot 2: Pairplot of All Parameters (Bivariate Relationships + KDE)
pairplot_fig = sns.pairplot(data, hue='species', diag_kind='kde', markers=["o", "s", "D"], height=2.2)
pairplot_fig.figure.suptitle("Pairwise Relationships of All Parameters by Species", y=1.02, fontsize=14)
pairplot_fig.savefig("eda_pairplot.png", dpi=300)
plt.close()
print("2. Pairwise feature plot saved as 'eda_pairplot.png'.")

# Visual Plot 3: Boxplots (Distribution & Outlier Analysis per Feature)
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()
for idx, col in enumerate(feature_cols):
    sns.boxplot(data=data, x='species', y=col, ax=axes[idx], hue='species', legend=False)
    axes[idx].set_title(f"Distribution of {col.replace('_', ' ').title()} by Species", fontsize=11)
    axes[idx].set_xlabel("Species")
    axes[idx].set_ylabel(f"{col} (cm)")
plt.suptitle("Feature Distributions and Outlier Identification", fontsize=14)
plt.tight_layout()
plt.savefig("eda_boxplots.png", dpi=300)
plt.close()
print("3. Feature boxplots saved as 'eda_boxplots.png'.")

# Visual Plot 4: Focused Parameter Relationships (Petal vs Sepal dimensions)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=data, x='petal_length', y='petal_width', hue='species',
                style='species', s=70, ax=ax1)
ax1.set_title("Petal Length vs. Petal Width\n(High Separation: r = 0.96)", fontsize=12)
ax1.set_xlabel("Petal Length (cm)")
ax1.set_ylabel("Petal Width (cm)")

sns.scatterplot(data=data, x='sepal_length', y='sepal_width', hue='species',
                style='species', s=70, ax=ax2)
ax2.set_title("Sepal Length vs. Sepal Width\n(Moderate Overlap: r = -0.11)", fontsize=12)
ax2.set_xlabel("Sepal Length (cm)")
ax2.set_ylabel("Sepal Width (cm)")

plt.tight_layout()
plt.savefig("eda_scatter_relationships.png", dpi=300)
plt.close()
print("4. Focused relationship scatter plots saved as 'eda_scatter_relationships.png'.")

# 8. 80/20 Random Train-Test Split
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

# 9. Summary of Parameters Considered
print("\n" + "="*55)
print("             PARAMETERS CONSIDERED")
print("="*55)
print("1. Feature / Input Parameters:")
print("   - sepal_length: Sepal length in cm (continuous numeric)")
print("   - sepal_width:  Sepal width in cm (continuous numeric)")
print("   - petal_length: Petal length in cm (continuous numeric)")
print("   - petal_width:  Petal width in cm (continuous numeric)")
print("\n2. Target / Output Parameter:")
print("   - species: Multiclass categorical label")
print("     Classes: Iris-setosa, Iris-versicolor, Iris-virginica")
print("\n3. Data Splitting Parameters:")
print("   - test_size: 0.20 (20% test data, 80% training data)")
print("   - random_state: 42 (ensures deterministic, reproducible split)")
print("   - stratify: y (ensures balanced class proportions across splits)")
print("\n4. Model Hyperparameters Considered:")
print("   [Logistic Regression]")
print("   - max_iter: 200 (maximum solver iterations to reach convergence)")
print("   - solver: 'lbfgs' (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)")
print("   - C: 1.0 (inverse regularization strength parameter)")
print("   - random_state: 42 (seed for solver consistency)")
print("   [Random Forest Classifier]")
print("   - n_estimators: 100 (number of decision trees in the ensemble)")
print("   - criterion: 'gini' (impurity measurement criterion)")
print("   - random_state: 42 (seed for tree building reproducibility)")
print("="*55)

# 10. Model Training on 80% Training Data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("\n--- Model Training on 80% Data ---")
# Train Logistic Regression
lr_model = LogisticRegression(max_iter=200, random_state=42)
lr_model.fit(X_train, y_train)
print(" Logistic Regression trained.")

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print(" Random Forest Classifier trained.")

# 11. Model Evaluation on 20% Test Data
models = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model
}

for model_name, model in models.items():
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro')
    rec_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    
    print("\n" + "="*55)
    print(f" EVALUATION RESULTS: {model_name.upper()} (on 20% Test Set)")
    print("="*55)
    print(f"Accuracy:                  {acc * 100:.2f}%")
    print(f"Precision (Macro Average): {prec_macro * 100:.2f}%")
    print(f"Recall (Macro Average):    {rec_macro * 100:.2f}%")
    print(f"F1-Score (Macro Average):  {f1_macro * 100:.2f}%")
    
    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in model.classes_],
                             columns=[f"Pred {c}" for c in model.classes_])
    print(cm_df)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

# 12. Test Set Actual vs Predicted Samples Comparison
print("\n--- Test Set Actual vs Predicted Comparison (First 10 Samples) ---")
test_comparison = X_test.copy()
test_comparison['Actual_Species'] = y_test
test_comparison['LR_Pred'] = lr_model.predict(X_test)
test_comparison['RF_Pred'] = rf_model.predict(X_test)
print(test_comparison.head(10))

# 13. Save Confusion Matrix Heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (model_name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_title(f"{model_name} Confusion Matrix\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    ax.set_xlabel("Predicted Species")
    ax.set_ylabel("Actual Species")

plt.tight_layout()
plt.savefig("model_evaluation_metrics.png", dpi=300)
plt.close()
print("\nConfusion matrix visualizations saved to 'model_evaluation_metrics.png'.")

