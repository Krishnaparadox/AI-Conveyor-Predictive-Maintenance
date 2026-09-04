import streamlit as st
import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh

st.markdown("""
<style>
    :root {
        --bg: #f5f7fb;
        --surface: #ffffff;
        --border: #e5e7eb;
        --text: #111827;
        --muted: #64748b;
        --accent: #2563eb;
        --success: #15803d;
        --warning: #b45309;
        --danger: #b91c1c;
    }

    .stApp {
        background: var(--bg);
    }

    .block-container {
        max-width: 1450px;
        padding: 2.2rem 2.5rem 3rem;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .hero {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 22px;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.05);
    }

    .hero-title {
        color: var(--text);
        font-size: 2.35rem;
        line-height: 1.1;
        font-weight: 750;
        letter-spacing: -0.04em;
        margin: 0;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        margin-top: 9px;
        max-width: 800px;
    }

    .section-heading {
        color: var(--text);
        font-size: 1.25rem;
        font-weight: 700;
        margin: 20px 0 12px;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 17px 18px;
        min-height: 112px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.035);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: var(--text);
        font-size: 1.65rem;
        font-weight: 720;
    }

    .status-card {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 9px;
        padding: 10px 12px;
        margin: 8px 0;
        color: #e5e7eb;
        font-size: 0.86rem;
    }

    .status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
    }

    .info-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.035);
    }

    .architecture-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px;
        min-height: 145px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.035);
    }

    .architecture-number {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .architecture-title {
        color: var(--text);
        font-size: 1rem;
        font-weight: 700;
        margin: 6px 0 8px;
    }

    .architecture-text {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.5;
    }

    .footer-text {
        text-align: center;
        color: #94a3b8;
        font-size: 0.78rem;
        padding-top: 12px;
    }

    .stButton > button {
        border-radius: 9px;
        font-weight: 650;
        min-height: 42px;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 14px;
    }
</style>
""", unsafe_allow_html=True)


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Conveyor AI Predictive Maintenance",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded"
)

# AUTO REFRESH

st_autorefresh(
    interval=3000,
    key="sensor_refresh"
)

# LOAD AI MODEL

model = joblib.load(
    "model/conveyor_model.pkl"
)

encoder = joblib.load(
    "model/fault_encoder.pkl"
)

# LOAD DATASET

data_path = "data/conveyor_data.csv"

dataset = pd.read_csv(
    data_path
)

# FEATURE NAMES

feature_names = [
    "Vibration",
    "Temperature",
    "Belt Speed",
    "Tension",
    "Motor Current",
    "Alignment"
]

fault_conditions = [
    "Normal",
    "Misalignment",
    "Overload",
    "Belt Damage",
    "Joint Failure"
]

# HEALTH SCORE

def calculate_health_score(fault):

    scores = {
        "Normal": 95,
        "Misalignment": 70,
        "Overload": 60,
        "Belt Damage": 40,
        "Joint Failure": 20
    }

    return scores.get(fault, 50)

# MAINTENANCE RECOMMENDATION

def maintenance_recommendation(fault):

    recommendations = {

        "Normal":
        "No immediate maintenance required. Continue monitoring.",

        "Misalignment":
        "Inspect belt alignment, rollers and tracking system.",

        "Overload":
        "Check material load and motor loading conditions.",

        "Belt Damage":
        "Inspect belt surface and damaged sections.",

        "Joint Failure":
        "Immediately inspect belt joint and schedule maintenance."
    }

    return recommendations.get(
        fault,
        "Perform detailed conveyor inspection."
    )

# MAINTENANCE PRIORITY

def maintenance_priority(fault):

    priorities = {

        "Normal": "LOW",
        "Misalignment": "MEDIUM",
        "Overload": "HIGH",
        "Belt Damage": "HIGH",
        "Joint Failure": "CRITICAL"
    }

    return priorities.get(
        fault,
        "MEDIUM"
    )

# FAILURE RISK

def failure_risk_score(fault, confidence):

    base_scores = {

        "Normal": 10,
        "Misalignment": 35,
        "Overload": 60,
        "Belt Damage": 75,
        "Joint Failure": 90
    }

    base_score = base_scores.get(
        fault,
        50
    )

    confidence_factor = confidence / 100

    risk_score = base_score * confidence_factor

    return round(risk_score)

# EARLY WARNING

def early_warning(risk_score):

    if risk_score >= 80:
        return "CRITICAL"

    elif risk_score >= 60:
        return "HIGH"

    elif risk_score >= 35:
        return "MEDIUM"

    else:
        return "LOW"

# DEGRADATION SCORE

def calculate_degradation(history_df):

    if len(history_df) < 5:

        return 0

    recent = history_df.tail(5)

    older = history_df.head(5)

    recent_vibration = recent["Vibration"].mean()

    older_vibration = older["Vibration"].mean()

    recent_temperature = recent["Temperature"].mean()

    older_temperature = older["Temperature"].mean()

    recent_current = recent["Motor Current"].mean()

    older_current = older["Motor Current"].mean()

    vibration_change = max(
        0,
        recent_vibration - older_vibration
    )

    temperature_change = max(
        0,
        recent_temperature - older_temperature
    )

    current_change = max(
        0,
        recent_current - older_current
    )

    score = (

        vibration_change * 8

        + temperature_change * 2

        + current_change * 0.5
    )

    return round(
        min(score, 100)
    )

# TITLE

st.markdown("""
<div class="hero">
    <div class="hero-title">Conveyor AI Predictive Maintenance</div>
    <div class="hero-subtitle">
        Multi-sensor condition monitoring, fault classification and
        maintenance decision support for industrial conveyor systems.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="dashboard-subtitle">AI-powered conveyor condition monitoring, fault detection and predictive maintenance decision support.</div>', unsafe_allow_html=True)

st.divider()

# SIDEBAR

st.sidebar.header(
    "️ System Controls"
)

scenario = st.sidebar.selectbox(

    "Operating Scenario",

    fault_conditions
)

if st.sidebar.button(
    "️ Reset Sensor History"
):

    st.session_state.sensor_history = []

    st.rerun()

st.sidebar.divider()

st.sidebar.subheader(
    "System Status"
)

st.sidebar.markdown('<div class="status-card"><span class="status-dot"></span>AI Model Online</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="status-card"><span class="status-dot"></span>Sensor Simulation Active</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="status-card">Refresh interval: 3 seconds</div>', unsafe_allow_html=True)

# SESSION STATE

if "sensor_history"not in st.session_state:

    st.session_state.sensor_history = []

# SENSOR DATA

scenario_data = dataset[
    dataset["fault_condition"] == scenario
]

if not scenario_data.empty:

    selected_row = scenario_data.sample(
        n=1
    ).iloc[0]

    vibration = selected_row["vibration"]

    temperature = selected_row["temperature"]

    belt_speed = selected_row["belt_speed"]

    tension = selected_row["tension"]

    motor_current = selected_row["motor_current"]

    alignment = selected_row["alignment"]

    # STORE HISTORY

    st.session_state.sensor_history.append({

        "Vibration": vibration,

        "Temperature": temperature,

        "Belt Speed": belt_speed,

        "Tension": tension,

        "Motor Current": motor_current,

        "Alignment": alignment
    })

    if len(
        st.session_state.sensor_history
    ) > 100:

        st.session_state.sensor_history = (
            st.session_state.sensor_history[-100:]
        )

    # CURRENT SENSOR READINGS

    st.header("Live Sensor Monitoring")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Vibration",
            f"{vibration:.2f}"
        )

        st.metric(
            "️ Temperature",
            f"{temperature:.2f}"
        )

    with col2:

        st.metric(
            "️ Belt Speed",
            f"{belt_speed:.2f}"
        )

        st.metric(
            "Tension",
            f"{tension:.2f}"
        )

    with col3:

        st.metric(
            "Motor Current",
            f"{motor_current:.2f}"
        )

        st.metric(
            "Alignment",
            f"{alignment:.2f}"
        )

    st.divider()

    # AI INPUT

    sensor_data = {

        "vibration": vibration,

        "temperature": temperature,

        "belt_speed": belt_speed,

        "tension": tension,

        "motor_current": motor_current,

        "alignment": alignment
    }

    data = pd.DataFrame([
        sensor_data
    ])

    # AI PREDICTION

    prediction = model.predict(
        data
    )

    fault = encoder.inverse_transform(
        prediction
    )[0]

    probabilities = model.predict_proba(
        data
    )[0]

    probability_map = dict(
        zip(
            encoder.classes_,
            probabilities
        )
    )

    confidence = (
        max(probabilities) * 100
    )

    # HEALTH AND RISK

    health_score = calculate_health_score(
        fault
    )

    risk_score = failure_risk_score(
        fault,
        confidence
    )

    warning_level = early_warning(
        risk_score
    )

    priority = maintenance_priority(
        fault
    )

    recommendation = maintenance_recommendation(
        fault
    )

    # AI SUMMARY

    st.header("AI Condition Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Detected Condition",
            fault
        )

    with col2:

        st.metric(
            "AI Confidence",
            f"{confidence:.1f}%"
        )

    with col3:

        st.metric(
            "Conveyor Health",
            f"{health_score}/100"
        )

    with col4:

        st.metric(
            "Failure Risk",
            f"{risk_score}/100"
        )

    st.divider()

    # MAINTENANCE STATUS

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Maintenance Priority")

        if priority == "CRITICAL":

            st.error(
                "CRITICAL — Immediate maintenance required."
            )

        elif priority == "HIGH":

            st.warning(
                "HIGH — Maintenance should be scheduled soon."
            )

        elif priority == "MEDIUM":

            st.warning(
                "MEDIUM — Inspect the conveyor system."
            )

        else:

            st.success(
                "LOW — System operating normally."
            )

    with col2:

        st.subheader(
            "️ Early Warning Level"
        )

        if warning_level == "CRITICAL":

            st.error(
                "CRITICAL FAILURE RISK"
            )

        elif warning_level == "HIGH":

            st.warning(
                "HIGH FAILURE RISK"
            )

        elif warning_level == "MEDIUM":

            st.warning(
                "MODERATE FAILURE RISK"
            )

        else:

            st.success(
                "LOW FAILURE RISK"
            )

    # DEGRADATION MONITOR

    st.divider()

    st.header("Conveyor Degradation Monitor")

    history_df = pd.DataFrame(
        st.session_state.sensor_history
    )

    degradation_score = calculate_degradation(
        history_df
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Degradation Score",
            f"{degradation_score}/100"
        )

    with col2:

        if degradation_score >= 70:

            degradation_status = "CRITICAL"

        elif degradation_score >= 45:

            degradation_status = "WARNING"

        elif degradation_score >= 20:

            degradation_status = "WATCH"

        else:

            degradation_status = "STABLE"

        st.metric(
            "Trend Status",
            degradation_status
        )

    with col3:

        st.metric(
            "Samples Collected",
            len(history_df)
        )

    if len(history_df) < 5:

        st.info(
            "Collecting sensor history... "
            "The degradation monitor becomes active after several readings."
        )

    elif degradation_score >= 70:

        st.error(
            "The sensor trend indicates significant degradation. "
            "Maintenance inspection is recommended."
        )

    elif degradation_score >= 45:

        st.warning(
            "The sensor trend indicates increasing operating stress. "
            "Continue monitoring and prepare maintenance."
        )

    else:

        st.success(
            "Sensor trends are currently stable."
        )

    # MAINTENANCE RECOMMENDATION

    st.divider()

    st.subheader("AI Maintenance Recommendation")

    st.info(
        recommendation
    )

    # FAULT PROBABILITY

    st.divider()

    st.subheader("AI Fault Probability")

    probability_df = pd.DataFrame({

        "Condition":
        list(probability_map.keys()),

        "Probability":
        [
            probability_map.get(
                condition,
                0
            ) * 100

            for condition in probability_map.keys()
        ]
    })

    probability_df = probability_df.sort_values(

        "Probability",

        ascending=False
    )

    st.bar_chart(

        probability_df.set_index(
            "Condition"
        )
    )

    # FEATURE IMPORTANCE

    st.subheader("AI Feature Importance")

    importance_df = pd.DataFrame({

        "Sensor":
        feature_names,

        "Importance":
        model.feature_importances_
    })

    importance_df = importance_df.sort_values(

        "Importance",

        ascending=False
    )

    st.bar_chart(

        importance_df.set_index(
            "Sensor"
        )
    )

    st.caption(
        "Higher importance means the AI model relies more heavily "
        "on that sensor when identifying conveyor conditions."
    )

# SENSOR HISTORY

if st.session_state.sensor_history:

    st.divider()

    st.header("Live Sensor Trends")

    history_df = pd.DataFrame(
        st.session_state.sensor_history
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Vibration")

        st.line_chart(
            history_df["Vibration"]
        )

        st.subheader(
            "️ Belt Speed"
        )

        st.line_chart(
            history_df["Belt Speed"]
        )

        st.subheader("Motor Current")

        st.line_chart(
            history_df["Motor Current"]
        )

    with col2:

        st.subheader(
            "️ Temperature"
        )

        st.line_chart(
            history_df["Temperature"]
        )

        st.subheader("Tension")

        st.line_chart(
            history_df["Tension"]
        )

        st.subheader("Alignment")

        st.line_chart(
            history_df["Alignment"]
        )

# SYSTEM ARCHITECTURE

st.divider()

st.header("System Architecture")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.subheader(
        "1️⃣ Sensors"
    )

    st.write(
        "Collect vibration, temperature, "
        "speed, tension, current and alignment data."
    )

with col2:

    st.subheader(
        "2️⃣ AI Model"
    )

    st.write(
        "Random Forest analyzes sensor patterns "
        "to classify conveyor conditions."
    )

with col3:

    st.subheader(
        "3️⃣ Risk Engine"
    )

    st.write(
        "AI results are converted into health, "
        "risk and degradation indicators."
    )

with col4:

    st.subheader(
        "4️⃣ Maintenance"
    )

    st.write(
        "The system recommends inspection "
        "or maintenance actions."
    )

# FOOTER

st.divider()

st.caption(
    "SIH Prototype • AI-based Conveyor Condition Monitoring • "
    "Predictive Maintenance Decision Support"
)
