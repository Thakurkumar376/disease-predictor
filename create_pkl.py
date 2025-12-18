import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# -------------------- Load and parse dataset --------------------
DATA_PATH = "dataset.csv"

df_raw = pd.read_csv(DATA_PATH, header=None)

# First column: target disease
df_raw[0] = df_raw[0].astype(str).str.strip()
y_text = df_raw[0]

# Other columns: symptom names (sparse, with NaN/empty)
symptom_cols = df_raw.columns[1:]

# Build global symptom list
symptoms = sorted(
    {
        s.strip()
        for col in symptom_cols
        for s in df_raw[col].astype(str).tolist()
        if str(s).strip() not in ("", "nan")
    }
)

# Map symptom to index for app.py
symptoms_dict = {symptom: idx for idx, symptom in enumerate(symptoms)}

# -------------------- Build multi-hot X matrix --------------------
def build_multi_hot(row, symptom_list, symptom_to_idx):
    vec = [0] * len(symptom_list)
    for col in symptom_cols:
        val = str(row[col]).strip()
        if not val or val.lower() == "nan":
            continue
        if val in symptom_to_idx:
            vec[symptom_to_idx[val]] = 1
    return vec

X_list = [build_multi_hot(row, symptoms, symptoms_dict) for _, row in df_raw.iterrows()]
X = pd.DataFrame(X_list, columns=symptoms)

# -------------------- Encode labels --------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y_text)

# -------------------- Train/validation split --------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -------------------- Model: RandomForest --------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

model.fit(X_train, y_train)

# -------------------- Evaluation --------------------
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)

print(f"✅ Training accuracy: {train_acc:.4f}")
print(f"✅ Validation accuracy: {val_acc:.4f}")
print(f"📊 Total diseases: {len(le.classes_)}")
print(f"🔢 Total symptoms: {len(symptoms)}")

# SAFE classification report - only classes present in validation set
val_classes = sorted(set(y_val))
val_class_names = [le.classes_[i] for i in val_classes]
print(f"\n📈 Validation report ({len(val_classes)}/{len(le.classes_)} classes):")
print(classification_report(y_val, y_val_pred, labels=val_classes, target_names=val_class_names))

# -------------------- Extra Dictionaries --------------------
description_dict = {disease: f"Description not available for {disease}."
                    for disease in le.classes_}

precaution_dict = {disease: ["No specific precaution information available."]
                   for disease in le.classes_}

# -------------------- Save bundle --------------------
bundle = {
    "decision_tree_model": model,
    "label_encoder": le,
    "symptoms_dict": symptoms_dict,
    "description_dict": description_dict,
    "precaution_dict": precaution_dict,
    "symptom_list": symptoms,
    "train_accuracy": train_acc,
    "val_accuracy": val_acc,
    "n_diseases": len(le.classes_),
    "n_symptoms": len(symptoms),
}

joblib.dump(bundle, "healthcare_model_bundle.pkl")
print("\n🎉 PKL file created successfully with RandomForest model!")
print("🚀 Ready to run your Flask app!")
