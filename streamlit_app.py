import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

#  FUNCTION: Return Selected Model
def get_model(model_name):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)

    elif model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=42)

    elif model_name == "XGBoost":
        if XGB_AVAILABLE:
            return XGBClassifier(
                eval_metric="logloss",
                use_label_encoder=False
            )
        else:
            st.warning("XGBoost not installed. Using Logistic Regression.")
            return LogisticRegression(max_iter=1000)

#  STREAMLIT UI
st.set_page_config(page_title="Loan ML App", layout="wide")
st.title("Loan Eligibility Prediction App")

st.sidebar.header("Model Settings")
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100
model_name = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Random Forest", "XGBoost"]
)

st.sidebar.header("Upload Data")
train_file = st.sidebar.file_uploader("Upload TRAIN CSV", type="csv")
test_file = st.sidebar.file_uploader("Upload TEST CSV", type="csv")

# MAIN LOGIC
if train_file is not None:

    data = pd.read_csv(train_file)

    st.subheader("Data Preview")
    st.dataframe(data.head())

       # Preprocessing
    data = data.drop("Loan_ID", axis=1)

    X = data.drop("Loan_Status", axis=1)
    y = data["Loan_Status"].map({"N": 0, "Y": 1})  # FIX for XGBoost

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X_train.select_dtypes(include=["object"]).columns

    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        encoders[col] = le

        # Model Training
    model = get_model(model_name)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluation
    st.subheader("Model Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.2%}")
        st.text("Classification Report")
        st.code(classification_report(y_test, y_pred))

    with col2:
        st.text("Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    # Test Predictions
    if test_file is not None:
        st.subheader("Test Predictions")

        test_data = pd.read_csv(test_file)
        loan_ids = test_data["Loan_ID"]

        test_data = test_data.drop("Loan_ID", axis=1)

        test_data[num_cols] = num_imputer.transform(test_data[num_cols])
        test_data[cat_cols] = cat_imputer.transform(test_data[cat_cols])

        for col in cat_cols:
            test_data[col] = encoders[col].transform(test_data[col])

        predictions = model.predict(test_data)

        result_df = pd.DataFrame({
            "Loan_ID": loan_ids,
            "Loan_Status_Prediction": predictions
        })

        result_df["Loan_Status_Prediction"] = result_df[
            "Loan_Status_Prediction"
        ].map({0: "N", 1: "Y"})

        st.dataframe(result_df.head())

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇Download Predictions",
            csv,
            "loan_predictions.csv",
            "text/csv"
        )

else:
    st.info("Upload training dataset to begin.")

