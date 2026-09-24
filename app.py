import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import joblib
model = joblib.load("models/customer_churn_model.pkl")
import os
# ============================================================
# SESSION STATE
# ============================================================

if "uploaded_dataset" not in st.session_state:
    st.session_state.uploaded_dataset = None
# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide"
)

# ---------------- SIDEBAR ---------------- #

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📂 Upload Dataset",
        "📊 Data Analysis",
        "🤖 Machine Learning",
        "🔮 Customer Prediction",
        "📑 Reports",
        "ℹ️ About"
    ]
)
# ---------------- HOME ---------------- #

if menu == "🏠 Home":

    st.title("📊 Customer Churn Prediction System")

    st.write("Enterprise Data Science & Machine Learning Project")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Customers", "1000")
    col2.metric("Features", "12")
    col3.metric("Model", "Random Forest")

    st.markdown("---")

    st.header("Project Description")

    st.write("""
    This application predicts customer churn using Machine Learning.

    ### Features

    - Upload Dataset
    - Data Analysis
    - Customer Churn Prediction
    - Random Forest Model
    - Interactive Dashboard
    """)

# ---------------- UPLOAD DATASET ---------------- #



elif menu == "📂 Upload Dataset":

    st.title("📂 Upload Customer Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)
            # Store uploaded dataset for other pages
            st.session_state.uploaded_dataset = df

            st.success("✅ Dataset uploaded successfully!")

            # Dataset Information
            st.subheader("📋 Dataset Information")

            col1, col2, col3 = st.columns(3)

            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", df.isnull().sum().sum())

            st.markdown("---")

            # Preview
            st.subheader("🔍 Dataset Preview")
            st.dataframe(df, use_container_width=True)

            # Column Names
            st.subheader("📌 Column Names")
            st.write(df.columns.tolist())

            # Data Types
            st.subheader("📊 Data Types")
            st.dataframe(df.dtypes.reset_index().rename(
                columns={"index":"Column", 0:"Data Type"}
            ))

            # Summary Statistics
            st.subheader("📈 Summary Statistics")
            st.dataframe(df.describe())

            # Missing Values
            st.subheader("❗ Missing Values")
            missing = pd.DataFrame({
                "Column": df.columns,
                "Missing Values": df.isnull().sum().values
            })

            st.dataframe(missing)

            # Duplicate Rows
            st.subheader("📄 Duplicate Records")

            duplicates = df.duplicated().sum()

            if duplicates == 0:
                st.success("No duplicate records found.")
            else:
                st.warning(f"{duplicates} duplicate rows found.")

        except Exception as e:
            st.error(f"Error reading file: {e}")

    else:
        st.info("Please upload a CSV dataset to continue.")

# ---------------- DATA ANALYSIS ---------------- #



elif menu == "📊 Data Analysis":

    st.title("📊 Enterprise Data Analysis Dashboard")

    uploaded_file = st.file_uploader(
        "Upload Customer Dataset",
        type=["csv"],
        key="eda"
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.success("Dataset Loaded Successfully")

        # ---------------- KPI ---------------- #

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing", df.isnull().sum().sum())
        col4.metric("Duplicates", df.duplicated().sum())

        st.divider()

        st.subheader("Dataset Preview")

        st.dataframe(df.head(10), use_container_width=True)

        st.divider()

        st.subheader("Summary Statistics")

        st.dataframe(df.describe())

        st.divider()

        st.subheader("Missing Values")

        st.dataframe(df.isnull().sum())

        st.divider()

        st.subheader("Correlation Matrix")

        numeric_df = df.select_dtypes(include=["number"])

        corr = numeric_df.corr()

        st.dataframe(corr)

        st.divider()

        st.subheader("Age Distribution")

        fig = px.histogram(
            df,
            x="Age",
            nbins=20,
            title="Age Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("Monthly Charges")

        fig = px.histogram(
            df,
            x="MonthlyCharges",
            nbins=20,
            title="Monthly Charges"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("Customer Churn")

        fig = px.pie(
            df,
            names="Churn",
            title="Customer Churn Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("Gender")

        fig = px.bar(
            df["Gender"].value_counts(),
            title="Gender Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("Contract Type")

        fig = px.bar(
            df["ContractType"].value_counts(),
            title="Contract Type"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("Internet Service")

        fig = px.bar(
            df["InternetService"].value_counts(),
            title="Internet Service"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(10, 6))

        cax = ax.imshow(corr)

        plt.colorbar(cax)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)

        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)

        st.pyplot(fig)
# ---------------- MACHINE LEARNING ---------------- #

elif menu == "🤖 Machine Learning":

    st.title("🤖 Machine Learning Dashboard")

    uploaded_file = st.file_uploader(
        "Upload Customer Dataset",
        type=["csv"],
        key="ml"
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.success("Dataset Loaded Successfully")

        # Encode categorical columns
        encoder = LabelEncoder()

        categorical_columns = [
            "Gender",
            "ContractType",
            "InternetService",
            "TechSupport",
            "PaymentMethod",
            "Churn"
        ]

        for col in categorical_columns:
            df[col] = encoder.fit_transform(df[col])

        # Features & Target
        X = df.drop("Churn", axis=1)
        y = df["Churn"]

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        if st.button("🚀 Train Model"):

            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            st.success("Model Trained Successfully")

            st.metric("Accuracy", f"{accuracy:.2%}")

            st.subheader("Classification Report")

            st.text(classification_report(y_test, y_pred))

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(5,5))

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm
            )

            disp.plot(ax=ax)

            st.pyplot(fig)

            # Feature Importance

            st.subheader("Feature Importance")

            importance = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            })

            importance = importance.sort_values(
                "Importance",
                ascending=False
            )

            st.dataframe(importance)

            fig = px.bar(
                importance,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # Save Model
            
            os.makedirs("outputs", exist_ok=True)
            output_file = "outputs/predictions.csv"
            if "prediction_history" in st.session_state:
                prediction_history = st.session_state["prediction_history"]

                if not prediction_history.empty:
                    prediction_history.to_csv(
                        output_file,
                        index=False
                    )

                    st.success(
                        "Prediction history saved successfully."
                    )


# ---------------- CUSTOMER PREDICTION ---------------- #

# ---------------- CUSTOMER PREDICTION ---------------- #

# ---------------- CUSTOMER PREDICTION ---------------- #

elif menu == "🔮 Customer Prediction":

    st.title("🔮 Customer Churn Prediction")

    customer_id = st.number_input("Customer ID", value=1001)
    age = st.number_input("Age", 18, 100, 35)

    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.number_input("Tenure", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", value=75.50)
    total_charges = st.number_input("Total Charges", value=900.00)

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic"]
    )

    tech = st.selectbox(
        "Tech Support",
        ["No", "Yes"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Credit Card",
            "Bank Transfer",
            "Electronic Check",
            "Cash"
        ]
    )

    usage = st.number_input("Monthly Usage (GB)", value=120)
    complaints = st.number_input("Complaints", value=1)

    if st.button("Predict Customer"):

        gender_value = 1 if gender == "Male" else 0

        contract_value = {
            "Month-to-month": 0,
            "One year": 1,
            "Two year": 2
        }[contract]

        internet_value = {
            "DSL": 0,
            "Fiber optic": 1
        }[internet]

        tech_value = {
            "No": 0,
            "Yes": 1
        }[tech]

        payment_value = {
            "Credit Card": 0,
            "Bank Transfer": 1,
            "Electronic Check": 2,
            "Cash": 3
        }[payment]

        new_customer = pd.DataFrame({
            "CustomerID": [customer_id],
            "Age": [age],
            "Gender": [gender_value],
            "Tenure": [tenure],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
            "ContractType": [contract_value],
            "InternetService": [internet_value],
            "TechSupport": [tech_value],
            "PaymentMethod": [payment_value],
            "MonthlyUsageGB": [usage],
            "NumComplaints": [complaints]
        })

        prediction = model.predict(new_customer)[0]

        if prediction == 1:
            result = "Churn"
            st.error("⚠️ Customer is likely to Churn")
        else:
            result = "Stay"
            st.success("✅ Customer is likely to Stay")

        st.markdown("---")
        st.subheader("Prediction Result")
        st.write(f"### Prediction: {result}")

        new_customer["Prediction"] = result

        st.markdown("---")
        st.subheader("Customer Details")
        st.dataframe(new_customer)

        os.makedirs("outputs", exist_ok=True)

        output_file = "outputs/predictions.csv"

# If the file already exists, append the new prediction
        if os.path.exists(output_file):
            old_predictions = pd.read_csv(output_file)
            all_predictions = pd.concat([old_predictions, new_customer], ignore_index=True)
        else:
            all_predictions = new_customer

# Save all predictions
        all_predictions.to_csv(output_file, index=False)

        st.success("Prediction saved successfully!")

        with open(output_file, "rb") as file:
            st.download_button(
                label="📥 Download Prediction CSV",
                data=file,
                file_name="predictions.csv",
                mime="text/csv"
            )

        
# ---------------- REPORTS ---------------- #

elif menu == "📑 Reports":

    st.title("📑 Customer Churn Reports")

    # Get uploaded dataset from session state
    df = st.session_state.uploaded_dataset

    if df is None:

        st.info(
            "📂 Please upload a customer dataset from the "
            "Upload Dataset page to generate reports."
        )

    else:

        st.success("✅ Report generated from the uploaded dataset.")

        # ============================================================
        # DATASET OVERVIEW
        # ============================================================

        st.subheader("📊 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Customers",
            df.shape[0]
        )

        col2.metric(
            "Total Features",
            df.shape[1]
        )

        col3.metric(
            "Missing Values",
            df.isnull().sum().sum()
        )

        col4.metric(
            "Duplicate Records",
            df.duplicated().sum()
        )

        st.markdown("---")

        # ============================================================
        # CHURN DISTRIBUTION
        # ============================================================

        if "Churn" in df.columns:

            st.subheader("📈 Customer Churn Distribution")

            churn_counts = (
                df["Churn"]
                .value_counts()
                .reset_index()
            )

            churn_counts.columns = [
                "Churn",
                "Customers"
            ]

            fig = px.pie(
                churn_counts,
                names="Churn",
                values="Customers",
                title="Customer Churn Distribution",
                hole=0.4
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                churn_counts,
                use_container_width=True
            )

        # ============================================================
        # NUMERIC FEATURES
        # ============================================================

        st.subheader("📊 Numeric Feature Summary")

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns

        if len(numeric_columns) > 0:

            numeric_summary = df[numeric_columns].describe().T

            numeric_summary = numeric_summary.reset_index()

            numeric_summary.rename(
                columns={
                    "index": "Feature"
                },
                inplace=True
            )

            st.dataframe(
                numeric_summary,
                use_container_width=True
            )

        else:

            st.info("No numeric features found in the dataset.")

        # ============================================================
        # TENURE ANALYSIS
        # ============================================================

        if "Tenure" in df.columns:

            st.subheader("📅 Customer Tenure Analysis")

            fig = px.histogram(
                df,
                x="Tenure",
                nbins=20,
                title="Customer Tenure Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ============================================================
        # MONTHLY CHARGES ANALYSIS
        # ============================================================

        if "MonthlyCharges" in df.columns:

            st.subheader("💰 Monthly Charges Analysis")

            fig = px.histogram(
                df,
                x="MonthlyCharges",
                nbins=20,
                title="Monthly Charges Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ============================================================
        # DATASET PREVIEW
        # ============================================================

        st.subheader("🔍 Report Data Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.success(
            "📑 Customer churn report generated successfully."
        )
    
# ---------------- ABOUT ---------------- #

elif menu == "ℹ️ About":

    st.title("About This Project")

    st.write("""
    ### Customer Churn Prediction System

    Technologies Used

    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    - Random Forest

    Developed as a Data Science Project.
    """)

    st.caption("© 2026 Customer Churn Prediction System")