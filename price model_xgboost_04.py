import numpy as np
import pandas as pd
import re
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# ✅ Load dataset
data = pd.read_csv('/Users/tonywall/Desktop/Service Seeking/XGBoost Model/Processed_with_scope_details.csv')

# ✅ Strip spaces from column names
data.columns = data.columns.str.strip()

# ✅ Print column names for debugging
print("Columns in dataset:", data.columns)

# ✅ Function to extract structured details dynamically from "scope"
def extract_property_type(text):
    property_types = ["house", "apartment", "commercial"]
    return next((p for p in property_types if p in str(text).lower()), "unknown")

def extract_storey(text):
    storey_keywords = ["one storey", "two storey", "three storey"]
    return next((s for s in storey_keywords if s in str(text).lower()), "unknown")

def extract_bedrooms(text):
    match = re.search(r'(\d+)-?bedroom', str(text))
    return int(match.group(1)) if match else 0  # Default to 0

def extract_size(text):
    match = re.search(r'(\d+)-?m', str(text))
    return int(match.group(1)) if match else np.nan

# ✅ Compute Smart Job Size dynamically
def assign_job_size(row):
    subtype, size, scope = row["Subtype"], row["Job_Size"], row["scope"]

    # Extract Property Type, Storey, Bedrooms from scope dynamically
    prop_type = extract_property_type(scope)
    storey = extract_storey(scope)
    bedrooms = extract_bedrooms(scope)

    if "Exterior home painting" in subtype:
        return size * (1.2 if prop_type == "house" else 0.8 if prop_type == "apartment" else 1.0)
    elif "Interior home painting" in subtype:
        storey_factor = 1.5 if "two storey" in storey else 2.0 if "three storey" in storey else 1.0
        return size * storey_factor * (1 + (bedrooms * 0.1))
    elif "Fences and fixtures" in subtype:
        return size * 2.5
    elif "Roof painting" in subtype:
        return size
    return size

# ✅ Step 1: Extract Job Size from scope
data["Job_Size"] = data["scope"].apply(extract_size)  # Assign first
data["Job_Size"].fillna(data["Job_Size"].median(), inplace=True)  # Then fill missing values

# ✅ Step 2: Compute Smart Job Size
data["Job_Size_Smart"] = data.apply(assign_job_size, axis=1)

# ✅ Step 3: Define function to adjust job size based on "Large" or "Small"
def adjust_job_size(row):
    """ Further adjusts Job Size based on size descriptor (Large, Small) in 'Subtype and Size' """
    job_size = row["Job_Size_Smart"]
    subtype_size = row["Subtype and Size"].lower()
    
    if "large" in subtype_size:
        return job_size * 1.3  # Increase by 30%
    elif "small" in subtype_size:
        return job_size * 0.7  # Decrease by 30%
    
    return job_size  # Default return unchanged

# ✅ Step 4: Apply adjustment once
data["Job_Size_Final"] = data.apply(adjust_job_size, axis=1)

# ✅ Step 5: Apply Log Transformation for Model Stability
data["Job_Size_Final_Log"] = np.log1p(data["Job_Size_Final"])  # Log(1 + x) to handle zero values

# ✅ Ensure text columns have no NaNs before vectorization
data["scope"] = data["scope"].fillna("")
data["updated_keywords"] = data["updated_keywords"].fillna("")

# ✅ Define Features & Target
X = data[["Subtype", "Subtype and Size", "updated_keywords", "scope", "Job_Size_Final_Log"]]
y = data["estimated_price"]

# ✅ Feature Engineering Pipeline (Dynamically Extracts Features)
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot_subtype", OneHotEncoder(handle_unknown="ignore"), ["Subtype"]),
        ("onehot_size", OneHotEncoder(handle_unknown="ignore"), ["Subtype and Size"]),
        ("tfidf_scope", TfidfVectorizer(max_features=200, stop_words="english"), "scope"),
        ("tfidf_keywords", TfidfVectorizer(max_features=100, stop_words="english"), "updated_keywords"),
        ("scaler", StandardScaler(), ["Job_Size_Final_Log"])
    ],
    remainder="drop"
)

# ✅ Apply Preprocessing
X_transformed = preprocessor.fit_transform(X)

# ✅ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# ✅ Train XGBoost Model
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=10,
    colsample_bytree=0.8,
    subsample=0.8,
    gamma=1,
    reg_lambda=3,
    reg_alpha=0.5,
    random_state=42
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# ✅ Evaluate Performance
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"🔹 XGBoost - MSE: {mse_xgb:.2f}")
print(f"🔹 XGBoost - R² Score: {r2_xgb:.4f}")

# ✅ Save Model & Preprocessor
joblib.dump(xgb_model, '/Users/tonywall/Desktop/Service Seeking/XGBoost Model/xgboost_model.pkl')
joblib.dump(preprocessor, '/Users/tonywall/Desktop/Service Seeking/XGBoost Model/preprocessor.pkl')

print("✅ Model and preprocessor saved for deployment.")

# ✅ Save Training Feature Names
feature_names = preprocessor.get_feature_names_out()
joblib.dump(feature_names, "/Users/tonywall/Desktop/Service Seeking/XGBoost Model/training_feature_names.pkl")

print(f"✅ Number of Training Features: {len(feature_names)}")

