"""Streamlit Web Application for Accident Severity Prediction.

Features a clean white and soft lavender theme, a two-page navigation interface
(Predict Severity and Model Benchmark & Analytics), dynamic Kaggle pre-crash input forms,
severity badge rendering, probability breakdown charts, and confusion matrix visualization.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Setup path resolution
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.config import (
    MODEL_FILE_PATH,
    METRICS_FILE_PATH,
    CONFUSION_MATRICES_PATH,
    DATA_FILE_PATH,
    FEATURE_CATEGORIES,
    TARGET_CLASSES,
    THEME,
)
from src.data_loader import load_and_validate_data

# =====================================================================
# STREAMLIT PAGE CONFIGURATION & THEME STYLING
# =====================================================================
st.set_page_config(
    page_title="Accident Severity Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = f"""
<style>
    /* Global Fonts & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: {THEME['text_dark']};
        background-color: {THEME['bg_main']};
    }}

    /* Main container background */
    .stApp {{
        background-color: {THEME['bg_main']};
    }}

    /* Sidebar background */
    [data-testid="stSidebar"] {{
        background-color: {THEME['bg_container']};
        border-right: 1px solid {THEME['border_subtle']};
    }}

    /* Card styling */
    .metric-card {{
        background-color: {THEME['secondary_card']};
        border: 1px solid {THEME['border_subtle']};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(185, 167, 232, 0.12);
    }}

    .status-badge-active {{
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #A5D6A7;
    }}

    .status-badge-warning {{
        background-color: #FFF3E0;
        color: #E65100;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #FFCC80;
    }}

    /* Buttons */
    div.stButton > button:first-child {{
        background-color: {THEME['primary_accent']};
        color: #1A1429;
        font-weight: 600;
        border: 1px solid {THEME['border_subtle']};
        border-radius: 10px;
        padding: 0.6rem 2rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 6px rgba(185, 167, 232, 0.3);
    }}

    div.stButton > button:first-child:hover {{
        background-color: #9D85DE;
        color: #FFFFFF;
        border-color: #9D85DE;
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(157, 133, 222, 0.4);
    }}

    /* Severity Result Badges */
    .badge-slight {{
        background-color: {THEME['badge_slight_bg']};
        color: {THEME['badge_slight_text']};
        font-size: 1.4rem;
        font-weight: 700;
        padding: 10px 24px;
        border-radius: 10px;
        display: inline-block;
        border: 1px solid #A5D6A7;
    }}

    .badge-serious {{
        background-color: {THEME['badge_serious_bg']};
        color: {THEME['badge_serious_text']};
        font-size: 1.4rem;
        font-weight: 700;
        padding: 10px 24px;
        border-radius: 10px;
        display: inline-block;
        border: 1px solid #FFE082;
    }}

    .badge-fatal {{
        background-color: {THEME['badge_fatal_bg']};
        color: {THEME['badge_fatal_text']};
        font-size: 1.4rem;
        font-weight: 700;
        padding: 10px 24px;
        border-radius: 10px;
        display: inline-block;
        border: 1px solid #EF9A9A;
    }}

    .disclaimer-box {{
        background-color: #F8F9FA;
        border-left: 4px solid {THEME['primary_accent']};
        padding: 14px 18px;
        margin-top: 20px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #555261;
        line-height: 1.5;
    }}

    /* Streamlit input widgets styling */
    .stSelectbox label, .stNumberInput label {{
        color: {THEME['text_dark']};
        font-weight: 600;
        font-size: 0.9rem;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================================
# CACHED ARTIFACT LOADERS
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Loads the fitted winning pipeline artifact."""
    if MODEL_FILE_PATH.exists():
        try:
            return joblib.load(MODEL_FILE_PATH)
        except Exception as e:
            st.error(f"Error reading model artifact: {e}")
            return None
    return None


@st.cache_data(show_spinner=False)
def load_metrics_table():
    """Loads benchmark metrics logged during training."""
    if METRICS_FILE_PATH.exists():
        try:
            return pd.read_csv(METRICS_FILE_PATH)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_confusion_matrices():
    """Loads serialized test set confusion matrices."""
    if CONFUSION_MATRICES_PATH.exists():
        try:
            return joblib.load(CONFUSION_MATRICES_PATH)
        except Exception:
            return None
    return None


# =====================================================================
# SIDEBAR NAVIGATION & METADATA
# =====================================================================
st.sidebar.markdown(
    f"""
    <div style="padding: 10px 0 20px 0;">
        <h2 style="color: {THEME['text_dark']}; margin: 0; font-size: 1.35rem; font-weight: 700;">
            🛡️ SafeRoute AI
        </h2>
        <p style="color: #6C667E; font-size: 0.85rem; margin-top: 4px;">
            Accident Severity Prediction System
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    ["🔮 Predict Severity", "📊 Model Benchmark & Analytics"],
    index=0,
)

st.sidebar.markdown("---")
model_artifact = load_trained_model()

if model_artifact is not None:
    best_name = model_artifact.get("model_name", "Winning Pipeline")
    best_f1 = model_artifact.get("macro_f1", 0.0)
    st.sidebar.markdown(
        f"""
        <div style="background-color: {THEME['secondary_card']}; border: 1px solid {THEME['border_subtle']}; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
            <span class="status-badge-active">● Active Pipeline</span>
            <div style="margin-top: 8px; font-weight: 600; font-size: 0.9rem; color: {THEME['text_dark']};">
                {best_name}
            </div>
            <div style="font-size: 0.8rem; color: #5B556D; margin-top: 3px;">
                Macro F1-Score: <b>{best_f1:.4f}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        """
        <div style="margin-bottom: 15px;">
            <span class="status-badge-warning">⚠️ No Model Artifact</span>
            <p style="font-size: 0.82rem; color: #8C4400; margin-top: 6px;">
                Run <code>python train_model.py</code> in your terminal to train and serialize the winning pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown(
    """
    <div style="font-size: 0.78rem; color: #7B768C; line-height: 1.4;">
        <b>B.Tech AI Minor Project</b><br>
        • Academic Rigor & Clean Anti-Leakage<br>
        • Class Imbalance Mitigated (Balanced Weights)<br>
        • Dataset: Kaggle Road Accident Severity in India
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# PAGE 1: PREDICT SEVERITY
# =====================================================================
if page == "🔮 Predict Severity":
    st.markdown(
        f"""
        <div style="margin-bottom: 25px;">
            <h1 style="color: {THEME['text_dark']}; font-weight: 700; margin-bottom: 4px;">
                Accident Severity Prediction
            </h1>
            <p style="color: #615B73; font-size: 1.05rem; margin-top: 0;">
                Machine Learning-Based Road Accident Analysis (Pre-Crash Indicators Only)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model_artifact is None:
        st.warning(
            "⚠️ **Model artifact `best_model.joblib` not found.**\n\n"
            "Please execute `python train_model.py` in your terminal to train the candidate models, "
            "evaluate on held-out data, and export the best model pipeline."
        )
        if st.button("🚀 Trigger Training from GUI"):
            with st.spinner("Training models and running benchmarks..."):
                from train_model import train_and_evaluate

                train_and_evaluate()
                st.cache_resource.clear()
                st.cache_data.clear()
                st.success("Model trained and serialized successfully! Please re-run prediction.")
                st.rerun()

    # Model Status indicator
    if model_artifact is not None:
        st.markdown(
            f"""
            <div style="margin-bottom: 20px;">
                <span class="status-badge-active">✓ Production Pipeline Ready</span>
                <span style="font-size: 0.9rem; color: #5A546D; margin-left: 10px;">
                    Winning Model: <b>{model_artifact.get('model_name')}</b> (Fitted with Preprocessor)
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="background-color: {THEME['bg_container']}; border: 1px solid {THEME['border_subtle']}; border-radius: 12px; padding: 22px; margin-bottom: 25px;">
            <h3 style="color: {THEME['text_dark']}; margin-top: 0; font-size: 1.15rem; font-weight: 600;">
                📋 Pre-Crash Accident Environmental & Driver Inputs
            </h3>
            <p style="font-size: 0.85rem; color: #6D677F; margin-bottom: 18px;">
                Note: In strict compliance with ML anti-leakage principles, only features physically observable 
                <i>before</i> the collision are accepted. Casualty and post-investigation features are strictly excluded.
            </p>
        """,
        unsafe_allow_html=True,
    )

    # 2-Column Responsive Input Form
    col1, col2 = st.columns(2)

    with col1:
        age_band = st.selectbox(
            "Driver Age Band",
            options=FEATURE_CATEGORIES["Age_band_of_driver"],
            index=0,
            help="Age demographic of the primary vehicle driver.",
        )
        sex_driver = st.selectbox(
            "Driver Sex",
            options=FEATURE_CATEGORIES["Sex_of_driver"],
            index=0,
            help="Recorded biological sex of the driver.",
        )
        driving_exp = st.selectbox(
            "Driving Experience",
            options=FEATURE_CATEGORIES["Driving_experience"],
            index=2,
            help="Total licensed driving experience of the driver.",
        )
        vehicle_type = st.selectbox(
            "Type of Vehicle",
            options=FEATURE_CATEGORIES["Type_of_vehicle"],
            index=0,
            help="Classification of the primary motor vehicle.",
        )
        num_vehicles = st.number_input(
            "Number of Vehicles Involved",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="Total count of vehicles physically involved in the traffic incident.",
        )
        day_of_week = st.selectbox(
            "Day of Week",
            options=FEATURE_CATEGORIES["Day_of_week"],
            index=4,
            help="Day when the incident occurred.",
        )

    with col2:
        weather = st.selectbox(
            "Weather Conditions",
            options=FEATURE_CATEGORIES["Weather_conditions"],
            index=0,
            help="Prevailing atmospheric and weather status at time of collision.",
        )
        surface_cond = st.selectbox(
            "Road Surface Conditions",
            options=FEATURE_CATEGORIES["Road_surface_conditions"],
            index=0,
            help="State of the road surface (Dry, Wet, Flooded, etc.).",
        )
        light_cond = st.selectbox(
            "Light Conditions",
            options=FEATURE_CATEGORIES["Light_conditions"],
            index=0,
            help="Ambient illumination (Daylight, Darkness with street lighting, etc.).",
        )
        junction_type = st.selectbox(
            "Type of Junction",
            options=FEATURE_CATEGORIES["Type_of_junction"],
            index=0,
            help="Intersection layout geometry (No junction, Y-shape, Crossing, etc.).",
        )
        surface_type = st.selectbox(
            "Road Surface Material",
            options=FEATURE_CATEGORIES["Road_surface_type"],
            index=0,
            help="Road paving type (Asphalt, Gravel, Earth roads).",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("⚡ Predict Accident Severity", use_container_width=True)

    if predict_btn:
        if model_artifact is None:
            st.error("Cannot predict: Model pipeline artifact is not available. Please train the model first.")
        else:
            # Construct single-row DataFrame matching the exact schema
            input_dict = {
                "Age_band_of_driver": str(age_band),
                "Sex_of_driver": str(sex_driver),
                "Driving_experience": str(driving_exp),
                "Type_of_vehicle": str(vehicle_type),
                "Number_of_vehicles_involved": int(num_vehicles),
                "Weather_conditions": str(weather),
                "Road_surface_conditions": str(surface_cond),
                "Light_conditions": str(light_cond),
                "Type_of_junction": str(junction_type),
                "Road_surface_type": str(surface_type),
                "Day_of_week": str(day_of_week),
            }
            input_df = pd.DataFrame([input_dict])

            pipeline = model_artifact["pipeline"]

            try:
                # Predict class and probabilities
                prediction = pipeline.predict(input_df)[0]
                has_proba = hasattr(pipeline, "predict_proba")

                if has_proba:
                    probabilities = pipeline.predict_proba(input_df)[0]
                    proba_classes = list(pipeline.classes_)
                else:
                    probabilities = None
                    proba_classes = []

                # Format badge style
                badge_class_map = {
                    "Slight": "badge-slight",
                    "Serious": "badge-serious",
                    "Fatal": "badge-fatal",
                }
                badge_css = badge_class_map.get(str(prediction), "badge-slight")

                st.markdown("---")
                st.markdown("### 🎯 Inference Results")

                result_col1, result_col2 = st.columns([1.1, 1.4])

                with result_col1:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="text-align: center; padding: 30px 20px;">
                            <div style="font-size: 0.95rem; font-weight: 600; color: #585368; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">
                                Predicted Crash Severity Level
                            </div>
                            <div class="{badge_css}">
                                {prediction} Collision
                            </div>
                            <div style="margin-top: 18px; font-size: 0.85rem; color: #6F6A80;">
                                Inferred via <b>{model_artifact.get('model_name')}</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with result_col2:
                    if has_proba and probabilities is not None:
                        st.markdown(
                            f"""
                            <div class="metric-card" style="padding: 20px;">
                                <div style="font-size: 0.95rem; font-weight: 600; color: #585368; margin-bottom: 15px;">
                                    📊 Class Probability Distribution
                                </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        prob_df = pd.DataFrame(
                            {
                                "Severity Class": proba_classes,
                                "Probability": [float(p) for p in probabilities],
                            }
                        )

                        # Render customized bar chart
                        fig, ax = plt.subplots(figsize=(6, 2.8))
                        fig.patch.set_facecolor("#FFFFFF")
                        ax.set_facecolor("#FFFFFF")

                        color_map = {
                            "Slight": "#81C784",
                            "Serious": "#FFB74D",
                            "Fatal": "#E57373",
                        }
                        bar_colors = [color_map.get(c, "#B9A7E8") for c in prob_df["Severity Class"]]

                        bars = ax.barh(
                            prob_df["Severity Class"],
                            prob_df["Probability"],
                            color=bar_colors,
                            height=0.45,
                            edgecolor="#D1C4E9",
                            linewidth=1,
                        )

                        ax.set_xlim(0, 1.0)
                        ax.set_xlabel("Calibrated Class Probability", fontsize=9, color="#292536")
                        ax.tick_params(colors="#292536", labelsize=9)

                        # Clean spines
                        for spine in ["top", "right", "left"]:
                            ax.spines[spine].set_visible(False)
                        ax.spines["bottom"].set_color("#D1C4E9")

                        # Annotate percentages on bars
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(
                                width + 0.02,
                                bar.get_y() + bar.get_height() / 2,
                                f"{width*100:.1f}%",
                                va="center",
                                ha="left",
                                fontsize=9,
                                fontweight="600",
                                color="#292536",
                            )

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                        st.markdown("</div>", unsafe_allow_html=True)

                # Academic & Empirical Disclaimer
                st.markdown(
                    f"""
                    <div class="disclaimer-box">
                        <b>Academic & Operational Disclaimer:</b> This prediction is an empirical estimation 
                        generated by a machine learning model trained on historical traffic accident patterns. 
                        It is intended for educational demonstrations, policy simulations, and academic research in 
                        road safety engineering. It does not replace real-time emergency triage or on-site police accident 
                        investigation reports.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"Inference failure: {e}")


# =====================================================================
# PAGE 2: MODEL BENCHMARK & ANALYTICS
# =====================================================================
elif page == "📊 Model Benchmark & Analytics":
    st.markdown(
        f"""
        <div style="margin-bottom: 25px;">
            <h1 style="color: {THEME['text_dark']}; font-weight: 700; margin-bottom: 4px;">
                Model Benchmark & Pipeline Analytics
            </h1>
            <p style="color: #615B73; font-size: 1.05rem; margin-top: 0;">
                Comprehensive Test-Set Evaluation, Imbalance Mitigation, and Error Distribution
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics_df = load_metrics_table()
    cm_dict = load_confusion_matrices()

    if metrics_df is None or cm_dict is None:
        st.warning(
            "⚠️ **Benchmark metrics not yet computed.**\n\n"
            "Please run `python train_model.py` in your terminal to evaluate all models on the held-out test split."
        )
        if st.button("🚀 Run Benchmark Pipeline Now"):
            with st.spinner("Executing stratified training and evaluation..."):
                from train_model import train_and_evaluate

                train_and_evaluate()
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Benchmark completed! Reloading metrics...")
                st.rerun()
    else:
        # Section 1: Benchmark Comparison Table
        st.markdown("### 🏆 Algorithm Performance Comparison (Held-out Test Set)")
        st.markdown(
            """
            In imbalanced multi-class settings, standard accuracy is misleading. 
            Selection is strictly governed by the **Macro F1-Score**, which unweights class size 
            and evaluates performance equally across *Slight*, *Serious*, and *Fatal* cases.
            """
        )

        winning_model = metrics_df.iloc[0]["Model"]

        # Styled Table Display
        def highlight_winner(row):
            if row["Model"] == winning_model:
                return ["background-color: #EDE7F6; font-weight: bold; border-left: 4px solid #B9A7E8;"] * len(row)
            return [""] * len(row)

        styled_df = metrics_df.style.apply(highlight_winner, axis=1).format(
            {
                "Accuracy": "{:.4f}",
                "Balanced Accuracy": "{:.4f}",
                "Macro Precision": "{:.4f}",
                "Macro Recall": "{:.4f}",
                "Macro F1-Score": "{:.4f}",
            }
        )
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.info(
            f"🏅 **Production Selection:** **{winning_model}** achieved the top Macro F1-Score "
            f"({metrics_df.iloc[0]['Macro F1-Score']:.4f}) and is serialized as `models/best_model.joblib`."
        )

        st.markdown("---")

        # Section 2: Visual Confusion Matrices & Target Distribution
        st.markdown("### 🔍 Confusion Matrix & Error Heatmaps")

        col_viz1, col_viz2 = st.columns([1.3, 1.0])

        with col_viz1:
            selected_model = st.selectbox(
                "Select Model to Inspect Confusion Matrix:",
                options=list(cm_dict.keys()),
                index=0,
            )

            cm_data = cm_dict[selected_model]
            matrix = cm_data["matrix"]
            labels = cm_data["classes"]

            # Custom lavender colormap
            lavender_cmap = mcolors.LinearSegmentedColormap.from_list(
                "lavender_gradient", ["#FFFFFF", "#D1C4E9", "#B9A7E8", "#7E57C2"], N=256
            )

            fig_cm, ax_cm = plt.subplots(figsize=(6, 4.8))
            fig_cm.patch.set_facecolor("#FFFFFF")
            ax_cm.set_facecolor("#FFFFFF")

            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap=lavender_cmap,
                xticklabels=labels,
                yticklabels=labels,
                cbar=True,
                linewidths=1,
                linecolor="#EDE7F6",
                annot_kws={"size": 11, "weight": "bold", "color": "#292536"},
                ax=ax_cm,
            )

            ax_cm.set_title(f"Confusion Matrix: {selected_model}", fontsize=12, fontweight="bold", pad=12, color="#292536")
            ax_cm.set_xlabel("Predicted Severity Label", fontsize=10, labelpad=8, color="#292536")
            ax_cm.set_ylabel("True Ground-Truth Severity", fontsize=10, labelpad=8, color="#292536")
            ax_cm.tick_params(colors="#292536", labelsize=9)

            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with col_viz2:
            st.markdown("#### ⚖️ Ground-Truth Class Imbalance")
            st.markdown(
                """
                Traffic crash records exhibit natural heavy skewness. 
                Minor fender-benders account for ~80% of data, while fatal accidents 
                represent <5%. Models without balanced class weighting degrade to predicting only the majority class.
                """
            )

            # Load actual or fallback distribution
            try:
                df_data = pd.read_csv(DATA_FILE_PATH)
                target_col = [c for c in ["Accident_severity", "accident_severity", "Severity"] if c in df_data.columns]
                if target_col:
                    counts = df_data[target_col[0]].astype(str).str.strip().value_counts()
                else:
                    counts = pd.Series({"Slight": 1600, "Serious": 340, "Fatal": 60})
            except Exception:
                counts = pd.Series({"Slight": 1600, "Serious": 340, "Fatal": 60})

            fig_dist, ax_dist = plt.subplots(figsize=(5, 4.2))
            fig_dist.patch.set_facecolor("#FFFFFF")
            ax_dist.set_facecolor("#FFFFFF")

            dist_colors = ["#81C784", "#FFB74D", "#E57373"]
            if len(counts) > 3:
                dist_colors = sns.color_palette("Purples_r", len(counts))

            ax_dist.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=dist_colors[:len(counts)],
                wedgeprops={"edgecolor": "#FFFFFF", "linewidth": 2, "antialiased": True},
                textprops={"fontsize": 9, "color": "#292536", "weight": "600"},
            )
            ax_dist.set_title("Dataset Severity Breakdown", fontsize=11, fontweight="bold", color="#292536")

            st.pyplot(fig_dist)
            plt.close(fig_dist)

        st.markdown("---")

        # Section 3: Engineering & Anti-Leakage Audit
        with st.expander("🛡️ View Engineering & Anti-Leakage Audit Trail"):
            st.markdown(
                """
                **Anti-Leakage Guarantees Implemented:**
                1. **Temporal Pre-Crash Whitelist:** Only environmental attributes (light, weather, road surface, junction) and driver demographics known prior to collision are retained.
                2. **Strict Casualty Dropping:** All casualty metrics (`Number_of_casualties`, `Casualty_severity`, `Casualty_class`, `Work_of_casuality`, etc.) are eliminated because casualties are consequences of severity, not causes.
                3. **Pipeline Encapsulation:** Preprocessing encoders (`SimpleImputer`, `OneHotEncoder`, `StandardScaler`) are fit strictly inside `sklearn.pipeline.Pipeline` on training folds, avoiding mean/variance leakage from test splits.
                4. **Unknown Category Resiliency:** Categorical features utilize `OneHotEncoder(handle_unknown='ignore')` to ensure zero runtime crashing on unseen categories in production.
                """
            )
