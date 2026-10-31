# ============================================================
# AURA
# Academic Understanding and Recommendation Assistant
#
# An Explainable Academic Pathway Recommendation System
# using XGBoost and SHAP
# ============================================================

# ------------------------------------------------------------
# IMPORT LIBRARIES
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="AURA",
    page_icon="🎓",
    layout="wide"
)

# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #F8FAFC;
}

div[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #E5E7EB;
    padding: 15px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# LOAD TRAINED MODEL AND ENCODERS
# ------------------------------------------------------------

model = joblib.load("models/xgboost_model.pkl")

interest_encoder = joblib.load(
    "models/interest_encoder.pkl"
)

career_encoder = joblib.load(
    "models/career_encoder.pkl"
)

pathway_encoder = joblib.load(
    "models/pathway_encoder.pkl"
)

# ------------------------------------------------------------
# CREATE SHAP EXPLAINER
# ------------------------------------------------------------

explainer = shap.TreeExplainer(model)

# ------------------------------------------------------------
# PAGE HEADER
# ------------------------------------------------------------

st.markdown("""
<h1 style='text-align:center;color:#1E3A8A;'>
🎓 AURA
</h1>

<h3 style='text-align:center;color:#4B5563;'>
Academic Understanding and Recommendation Assistant
</h3>

<p style='text-align:center;color:#6B7280;'>
An Explainable Academic Pathway Recommendation System
Using XGBoost and SHAP
</p>
""", unsafe_allow_html=True)

st.divider()

# ------------------------------------------------------------
# SIDEBAR INPUT
# ------------------------------------------------------------

st.sidebar.header("📚 Student Profile")

gpa = st.sidebar.number_input(
    "GPA",
    min_value=1.00,
    max_value=5.00,
    value=2.00,
    step=0.01
)

programming = st.sidebar.slider(
    "Programming Grade",
    60,
    100,
    85
)

mathematics = st.sidebar.slider(
    "Mathematics Grade",
    60,
    100,
    85
)

statistics = st.sidebar.slider(
    "Statistics Grade",
    60,
    100,
    85
)

interest = st.sidebar.selectbox(
    "Interest Area",
    interest_encoder.classes_
)

career = st.sidebar.selectbox(
    "Career Goal",
    career_encoder.classes_
)

generate = st.sidebar.button(
    "🚀 Generate Recommendation"
)

# ------------------------------------------------------------
# PROCESS WHEN BUTTON IS CLICKED
# ------------------------------------------------------------

if generate:

    # --------------------------------------------------------
    # PREPARE INPUT DATA
    # --------------------------------------------------------

    input_data = pd.DataFrame([

        {
            "GPA": gpa,

            "Programming": programming,

            "Mathematics": mathematics,

            "Statistics": statistics,

            "Interest":
                interest_encoder.transform(
                    [interest]
                )[0],

            "CareerGoal":
                career_encoder.transform(
                    [career]
                )[0]
        }

    ])

    # --------------------------------------------------------
    # XGBOOST PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_data)

    pathway = pathway_encoder.inverse_transform(
        prediction
    )[0]

    probabilities = model.predict_proba(
        input_data
    )

    confidence = (
        probabilities.max() * 100
    )

    # --------------------------------------------------------
    # READINESS SCORE
    # --------------------------------------------------------

    readiness = (

        programming

        + mathematics

        + statistics

    ) / 3

    # --------------------------------------------------------
    # SHAP COMPUTATION
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        input_data
    )

    predicted_class = int(prediction[0])

    # --------------------------------------------------------
    # ROBUST SHAP HANDLING
    # Supports:
    # - Binary Models
    # - Multi-class Models
    # - Older SHAP Versions
    # --------------------------------------------------------

    try:

        if isinstance(shap_values, list):

            contributions = (
                shap_values[predicted_class][0]
            )

        elif len(shap_values.shape) == 3:

            contributions = (
                shap_values[
                    0,
                    :,
                    predicted_class
                ]
            )

        elif len(shap_values.shape) == 2:

            contributions = shap_values[0]

        else:

            contributions = (
                shap_values.flatten()
            )

    except Exception:

        contributions = [0] * len(
            input_data.columns
        )

    # --------------------------------------------------------
    # BUILD FEATURE IMPORTANCE TABLE
    # --------------------------------------------------------

    importance_df = pd.DataFrame({

        "Feature":
            list(input_data.columns),

        "Contribution":
            list(contributions)

    })

    importance_df["AbsoluteImpact"] = (

        importance_df["Contribution"]

        .abs()

    )

    importance_df = (

        importance_df

        .sort_values(

            by="AbsoluteImpact",

            ascending=False

        )

    )

    top_features = (
        importance_df.head(5)
    )

    # --------------------------------------------------------
    # USER INTERFACE TABS
    # --------------------------------------------------------

    tab1, tab2, tab3 = st.tabs([

        "🎯 Recommendation",

        "📊 Analytics",

        "🔍 Explainability"

    ])

    # ========================================================
    # TAB 1 : RECOMMENDATION
    # ========================================================

    with tab1:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Academic Readiness",
            f"{readiness:.1f}%"
        )

        col2.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        col3.metric(
            "Recommended Pathway",
            pathway
        )

        st.success(
            f"🎓 Recommended Pathway: {pathway}"
        )

        strongest_feature = (
            top_features.iloc[0]["Feature"]
        )

        second_feature = (
            top_features.iloc[1]["Feature"]
        )

        st.info(

            f"""
            AURA recommended
            **{pathway}**
            because the student's

            **{strongest_feature}**
            and
            **{second_feature}**

            had the strongest
            influence on the
            prediction generated
            by the XGBoost model.
            """
        )

        career_map = {

            "Artificial Intelligence":
                "AI Engineer, Machine Learning Engineer, AI Researcher",

            "Data Science":
                "Data Analyst, Data Scientist, Business Intelligence Analyst",

            "Cybersecurity":
                "Security Analyst, SOC Analyst, Penetration Tester",

            "Software Engineering":
                "Software Developer, Full Stack Developer, Solutions Architect",

            "Information Systems":
                "Systems Analyst, IT Manager, Project Coordinator"

        }

        st.subheader(
            "🚀 Suggested Career Opportunities"
        )

        st.success(
            career_map.get(
                pathway,
                "Technology Professional"
            )
        )

    # ========================================================
    # TAB 2 : ANALYTICS
    # ========================================================

    with tab2:

        st.subheader(
            "📊 Academic Performance Profile"
        )

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        subjects = [

            "Programming",

            "Mathematics",

            "Statistics"

        ]

        scores = [

            programming,

            mathematics,

            statistics

        ]

        colors = [

            "#2563EB",

            "#10B981",

            "#F59E0B"

        ]

        ax.bar(
            subjects,
            scores,
            color=colors
        )

        ax.set_ylim(0, 100)

        ax.set_ylabel("Score")

        ax.set_title(
            "Student Academic Performance"
        )

        st.pyplot(fig)

    # ========================================================
    # TAB 3 : SHAP EXPLAINABILITY
    # ========================================================

    with tab3:

        st.subheader(
            "🔍 Explainable AI Insights"
        )

        st.write("""
        AURA uses SHAP
        (SHapley Additive Explanations)
        to determine which factors had
        the greatest influence on the
        recommendation.
        """)

        st.divider()

        st.subheader(
            "Top Recommendation Factors"
        )

        for _, row in top_features.iterrows():

            contribution = row[
                "Contribution"
            ]

            if contribution >= 0:

                st.success(
                    f"⬆️ {row['Feature']} "
                    f"({contribution:.4f})"
                )

            else:

                st.error(
                    f"⬇️ {row['Feature']} "
                    f"({contribution:.4f})"
                )

        st.divider()

        st.subheader(
            "📈 SHAP Feature Importance"
        )

        fig2, ax2 = plt.subplots(
            figsize=(8, 4)
        )

        colors = [

            "#10B981"
            if value >= 0
            else "#EF4444"

            for value in top_features[
                "Contribution"
            ]

        ]

        ax2.barh(

            top_features["Feature"],

            top_features["Contribution"],

            color=colors

        )

        ax2.set_xlabel(
            "Contribution Strength"
        )

        ax2.set_title(
            "Feature Influence on Recommendation"
        )

        st.pyplot(fig2)

        st.divider()

        st.subheader(
            "Feature Importance Summary"
        )

        st.dataframe(
            top_features[
                [
                    "Feature",
                    "Contribution"
                ]
            ],
            use_container_width=True
        )

        st.info("""
        Positive SHAP values increased the
        likelihood of the recommended pathway,
        while negative values reduced its
        likelihood. These explanations improve
        transparency and help users understand
        the recommendation process.
        """)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "AURA | Academic Understanding and Recommendation Assistant | Powered by Python, XGBoost, SHAP, and Streamlit"
)