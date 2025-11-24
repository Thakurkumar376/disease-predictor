import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------- Symptoms --------------------
symptoms = [
    "fever", "cough", "headache", "fatigue", "sore_throat"
]

symptoms_dict = {symptom: idx for idx, symptom in enumerate(symptoms)}

# -------------------- Disease Data --------------------
data = {
    "fever": [1, 0, 1, 0, 1],
    "cough": [0, 1, 1, 0, 1],
    "headache": [1, 0, 0, 1, 0],
    "fatigue": [0, 1, 1, 0, 1],
    "sore_throat": [1, 0, 0, 1, 1],
    "disease": ["Flu", "Cold", "Flu", "Migraine", "Flu"]
}

df = pd.DataFrame(data)
X = df.drop("disease", axis=1)
y = df["disease"]

# -------------------- Encoding --------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -------------------- Model --------------------
model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# -------------------- Extra Dictionaries --------------------
description_dict = {
    "Flu": "Influenza is a common viral infection.",
    "Cold": "Common cold is a mild viral respiratory illness.",
    "Migraine": "Migraine is a neurological condition causing headaches."
}

precaution_dict = {
    "Flu": ["Drink fluids", "Take rest", "Consult doctor"],
    "Cold": ["Stay warm", "Drink warm fluids"],
    "Migraine": ["Rest in dark room", "Avoid loud noise"]
}

# -------------------- Save Bundle --------------------
bundle = {
    "decision_tree_model": model,
    "label_encoder": le,
    "symptoms_dict": symptoms_dict,
    "description_dict": description_dict,
    "precaution_dict": precaution_dict
}

joblib.dump(bundle, "healthcare_model_bundle.pkl")
print("✅ PKL file created successfully with all keys!")
