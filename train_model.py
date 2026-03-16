import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

print("Đọc dữ liệu...")

train = pd.read_csv("train.csv")

# =============================
# CHỌN FEATURE
# =============================

numeric_features = [
    "Count_F",
    "Tuition_Debt",
    "Training_Score_Mixed",
    "Age"
]

categorical_features = [
    "Club_Member",
    "English_Level"
]

features = numeric_features + categorical_features

X = train[features]
y = train["Academic_Status"]

# =============================
# CHIA DATA
# =============================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =============================
# PREPROCESS
# =============================

preprocess = ColumnTransformer([

    ("num",
     SimpleImputer(strategy="median"),
     numeric_features),

    ("cat",
     OneHotEncoder(handle_unknown="ignore"),
     categorical_features)

])

# =============================
# MODEL
# =============================

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ))
])

print("Huấn luyện model...")

model.fit(X_train, y_train)

preds = model.predict(X_val)

print("\nĐánh giá model:")
print(classification_report(y_val, preds))

# =============================
# SAVE MODEL
# =============================

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Đã lưu model.pkl")