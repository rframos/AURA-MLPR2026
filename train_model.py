import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load Dataset
df = pd.read_csv("data/student_pathways.csv")

# Encode categorical fields
le_interest = LabelEncoder()
le_career = LabelEncoder()
le_pathway = LabelEncoder()

df["Interest"] = le_interest.fit_transform(df["Interest"])
df["CareerGoal"] = le_career.fit_transform(df["CareerGoal"])
df["Pathway"] = le_pathway.fit_transform(df["Pathway"])

# Features
X = df.drop("Pathway", axis=1)

# Target
y = df["Pathway"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train XGBoost
model = XGBClassifier(
    objective="multi:softmax",
    num_class=len(le_pathway.classes_),
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# Test Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Save Model
joblib.dump(model, "models/xgboost_model.pkl")
joblib.dump(le_interest, "models/interest_encoder.pkl")
joblib.dump(le_career, "models/career_encoder.pkl")
joblib.dump(le_pathway, "models/pathway_encoder.pkl")

print("Model Saved Successfully")