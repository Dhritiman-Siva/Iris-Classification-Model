import joblib
import os
import pandas as pd
from sklearn.tree import export_text

model_path = 'iris_classifier_model.joblib'

print("=" * 65)
print("       INSPECTING CONTENT OF: " + model_path)
print("=" * 65)

if not os.path.exists(model_path):
    print(f"Error: '{model_path}' not found in current directory.")
    exit(1)

# 1. Load the binary model
model = joblib.load(model_path)
file_size_kb = os.path.getsize(model_path) / 1024

print(f"\n[1] FILE METADATA")
print(f"    - File Name     : {model_path}")
print(f"    - File Size     : {file_size_kb:.2f} KB")
print(f"    - Python Type   : {type(model).__module__}.{type(model).__name__}")

print(f"\n[2] TARGET CLASSES & FEATURES STORED")
print(f"    - Target Classes ({len(model.classes_)}): {list(model.classes_)}")
print(f"    - Input Features ({model.n_features_in_}): {list(model.feature_names_in_)}")

print(f"\n[3] MODEL HYPERPARAMETERS")
params = model.get_params()
key_params = [
    'n_estimators', 'criterion', 'max_depth', 'min_samples_split',
    'min_samples_leaf', 'bootstrap', 'random_state', 'n_jobs'
]
for p in key_params:
    print(f"    - {p.ljust(18)}: {params.get(p)}")

print(f"\n[4] LEARNED FEATURE IMPORTANCES")
feature_imp = pd.DataFrame({
    'Feature': model.feature_names_in_,
    'Weight': (model.feature_importances_ * 100).round(2)
}).sort_values(by='Weight', ascending=False)
for _, row in feature_imp.iterrows():
    bar = "#" * int(row['Weight'] / 2)
    print(f"    - {row['Feature'].ljust(14)}: {row['Weight']:5.2f}%  {bar}")

print(f"\n[5] INTERNAL DECISION TREE STRUCTURE (Tree 1 of {model.n_estimators})")
print("    Below is the exact decision logic inside one of the 100 trained trees:")
print("-" * 65)
tree_rules = export_text(model.estimators_[0], feature_names=list(model.feature_names_in_))
# Indent the tree rules for neat display
for line in tree_rules.strip().split("\n"):
    print("    " + line)
print("-" * 65)
print("    Note: class 0.0 = Iris-setosa, class 1.0 = Iris-versicolor, class 2.0 = Iris-virginica")

print(f"\n[6] TEST PREDICTION USING THE LOADED JOBLIB OBJECT")
test_sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=model.feature_names_in_)
predicted_species = model.predict(test_sample)[0]
confidence = max(model.predict_proba(test_sample)[0]) * 100

print(f"    - Input  : Sepal (5.1 x 3.5 cm), Petal (1.4 x 0.2 cm)")
print(f"    - Output : {predicted_species} ({confidence:.1f}% Confidence)")
print("=" * 65)
