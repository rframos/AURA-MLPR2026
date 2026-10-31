import shap
import joblib
import pandas as pd

# Load model
model = joblib.load("models/xgboost_model.pkl")

# Load data
df = pd.read_csv("data/student_pathways.csv")

X = df.drop("Pathway", axis=1)

# Encode categorical fields
X["Interest"] = X["Interest"].astype("category").cat.codes
X["CareerGoal"] = X["CareerGoal"].astype("category").cat.codes

# SHAP Explainer
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

# Generate Summary Plot
shap.summary_plot(
    shap_values,
    X
)