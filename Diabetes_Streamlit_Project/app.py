
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_diabetes


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Progression Prediction",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# LOAD FINAL MODEL
# =========================================================

model = joblib.load("diabetes_model.pkl")


# =========================================================
# LOAD DATASET
# =========================================================

diabetes = load_diabetes(as_frame=True)
df = diabetes.frame


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
}

.subtitle {
    font-size: 18px;
    margin-bottom: 15px;
}

.tagline {
    background-color: #5427A8;
    padding: 10px 18px;
    border-radius: 12px;
    display: inline-block;
    font-size: 16px;
    font-weight: 600;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
}

.info-card {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #444;
}

.baseline-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #D9534F;
}

.best-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #28A745;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🩺 Navigation")

page = st.sidebar.radio(
    "Explore Project",
    [
        "🏠 Home",
        "📊 Dataset Insights",
        "🔮 Prediction",
        "🏆 Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.write(
    "🧠 Machine Learning Regression"
)

st.sidebar.write(
    "🏆 Final Model: Gradient Boosting"
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    col1, col2 = st.columns([1.2, 1])

    with col1:

        st.markdown(
            '<div class="main-title">'
            '🩺 Diabetes Progression Prediction'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="subtitle">'
            'Smart Machine Learning Application for '
            'Predicting Diabetes Disease Progression'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="tagline">'
            '✨ Analyze • Optimize • Predict'
            '</div>',
            unsafe_allow_html=True
        )

    with col2:

        st.info(
            "🩺 Diabetes • Machine Learning • Prediction"
        )

    st.divider()


    # -----------------------------------------------------
    # PROJECT METRICS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "👥 Dataset Records",
            "442"
        )

    with col2:

        st.metric(
            "🧬 Input Features",
            "10"
        )

    with col3:

        st.metric(
            "🏆 Final Model",
            "Gradient Boosting"
        )


    st.write("")

    st.markdown(
        '<div class="section-title">'
        '🎯 Project Objective'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Predict the quantitative diabetes disease-progression "
        "score using patient baseline features and machine "
        "learning regression techniques."
    )


    # -----------------------------------------------------
    # MODEL JOURNEY
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📈 Model Journey'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="baseline-card">

            ### 🔵 Baseline Model

            **Linear Regression**

            Simple and interpretable
            starting model.

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="best-card">

            ### 🚀 Optimized Model

            **Gradient Boosting**

            Selected after hyperparameter
            tuning and model comparison.

            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")

    st.success(
        "🏆 The final model was selected based on the "
        "highest R² score from the model comparison."
    )


# =========================================================
# DATASET INSIGHTS
# =========================================================

elif page == "📊 Dataset Insights":

    st.title("📊 Dataset Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("👥 Records", df.shape[0])

    with col2:
        st.metric("🧬 Features", 10)

    with col3:
        st.metric("🎯 Target", "Disease Progression")

    st.divider()

    st.subheader("🔍 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("🧬 Feature Information")

    feature_info = pd.DataFrame({

        "Feature": [
            "age",
            "sex",
            "bmi",
            "bp",
            "s1",
            "s2",
            "s3",
            "s4",
            "s5",
            "s6"
        ],

        "Display Name": [
            "👤 Age",
            "⚧️ Sex",
            "⚖️ Body Mass Index",
            "❤️ Blood Pressure",
            "🧪 Total Cholesterol",
            "🧬 LDL",
            "💙 HDL",
            "📊 Cholesterol / HDL",
            "🧪 Triglycerides",
            "🩸 Blood Sugar"
        ]
    })

    st.dataframe(
        feature_info,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PREDICTION
# =========================================================

elif page == "🔮 Prediction":

    st.title("🔮 Diabetes Progression Prediction")

    st.write(
        "Enter the feature values used by the trained "
        "machine learning model."
    )

    st.warning(
        "⚠️ The Scikit-learn Diabetes Dataset uses "
        "standardized feature values. Enter values using "
        "the same scaled format as the dataset."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("👤 Patient Information")

        age = st.number_input(
            "👤 Age",
            value=0.0,
            format="%.5f"
        )

        sex = st.number_input(
            "⚧️ Sex",
            value=0.0,
            format="%.5f"
        )

        bmi = st.number_input(
            "⚖️ Body Mass Index",
            value=0.0,
            format="%.5f"
        )

        bp = st.number_input(
            "❤️ Blood Pressure",
            value=0.0,
            format="%.5f"
        )

        s1 = st.number_input(
            "🧪 Total Cholesterol (s1)",
            value=0.0,
            format="%.5f"
        )

    with col2:

        st.subheader("🧪 Blood Measurements")

        s2 = st.number_input(
            "🧬 LDL (s2)",
            value=0.0,
            format="%.5f"
        )

        s3 = st.number_input(
            "💙 HDL (s3)",
            value=0.0,
            format="%.5f"
        )

        s4 = st.number_input(
            "📊 Cholesterol / HDL (s4)",
            value=0.0,
            format="%.5f"
        )

        s5 = st.number_input(
            "🧪 Triglycerides (s5)",
            value=0.0,
            format="%.5f"
        )

        s6 = st.number_input(
            "🩸 Blood Sugar (s6)",
            value=0.0,
            format="%.5f"
        )


    st.divider()


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    if st.button(
        "🚀 Generate Prediction",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [[
                age,
                sex,
                bmi,
                bp,
                s1,
                s2,
                s3,
                s4,
                s5,
                s6
            ]],
            columns=[
                "age",
                "sex",
                "bmi",
                "bp",
                "s1",
                "s2",
                "s3",
                "s4",
                "s5",
                "s6"
            ]
        )

        prediction = model.predict(input_data)[0]

        st.success(
            "✅ Prediction Generated Successfully!"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🔮 Predicted Disease Progression",
                f"{prediction:.2f}"
            )

        with col2:

            st.metric(
                "🤖 Model Used",
                "Gradient Boosting"
            )

        st.info(
            "ℹ️ This application is intended for educational "
            "and machine-learning demonstration purposes "
            "and is not a medical diagnostic tool."
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "🏆 Model Performance":

    st.title("🏆 Model Performance")

    st.write(
        "Comparison of the machine learning models evaluated "
        "during the project."
    )

    st.success(
        "🏆 Final model selected from the highest R² score."
    )

    # Use the actual results created earlier
    results_df = pd.read_csv("model_results.csv")
    display_results = results_df.copy()

    display_results["R² Percentage"] = (
        display_results["R² Score"] * 100
    ).round(2).astype(str) + "%"

    st.subheader("📊 Model Comparison")

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📈 R² Score Comparison")

    chart_data = (
        results_df
        .set_index("Model")["R² Score"]
        * 100
    )

    st.bar_chart(chart_data)

    st.subheader("💡 Model Evaluation")

    st.write(
        """
        **R² Score:** Higher values indicate better
        explanatory performance.

        **MAE:** Lower values indicate smaller average
        prediction errors.

        **RMSE:** Lower values indicate better control
        over larger prediction errors.
        """
    )
