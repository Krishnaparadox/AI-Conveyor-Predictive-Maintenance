import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Conveyor Health Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme-aware dashboard styling

st.markdown("""
<style>

.stApp {
    background-color: var(--background-color);
    color: var(--text-color);
}

[data-testid="stAppViewContainer"] {
    background-color: var(--background-color);
    color: var(--text-color);
}

[data-testid="stAppViewContainer"] main {
    background-color: var(--background-color);
    color: var(--text-color);
}

[data-testid="stSidebar"] {
    background-color: var(--secondary-background-color);
    color: var(--text-color);
    border-right: 1px solid rgba(128,128,128,0.20);
}

[data-testid="stSidebar"] * {
    color: var(--text-color);
}

.block-container {
    max-width: 1480px;
    padding: 1.8rem 2.25rem 3rem;
}

.hero {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 22px;
    padding: 30px 34px;
    margin-bottom: 24px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
}

.hero-kicker {
    color: #2f7df6;
    font-size: 0.73rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
}

.hero-title {
    color: var(--text-color);
    font-size: 2.45rem;
    line-height: 1.1;
    font-weight: 780;
    letter-spacing: -0.04em;
    margin: 0;
}

.hero-subtitle {
    color: var(--text-color);
    opacity: 0.70;
    max-width: 850px;
    font-size: 0.96rem;
    line-height: 1.6;
    margin-top: 10px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-top: 17px;
    background: rgba(24,167,102,0.10);
    border: 1px solid rgba(24,167,102,0.25);
    color: #18a766;
    border-radius: 999px;
    padding: 7px 11px;
    font-size: 0.76rem;
    font-weight: 700;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #18a766;
    box-shadow: 0 0 0 4px rgba(24,167,102,0.12);
}

.sidebar-brand-title {
    color: var(--text-color);
    font-size: 1rem;
    font-weight: 750;
}

.sidebar-brand-subtitle {
    color: var(--text-color);
    opacity: 0.60;
    font-size: 0.73rem;
    margin-top: 3px;
}

.status-card {
    display: flex;
    align-items: center;
    gap: 9px;
    background-color: var(--background-color);
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 10px;
    padding: 10px 11px;
    margin: 7px 0;
    font-size: 0.81rem;
    color: var(--text-color);
}

.section-title {
    color: var(--text-color);
    font-size: 1.16rem;
    font-weight: 730;
    letter-spacing: -0.02em;
    margin: 0;
}

.section-subtitle {
    color: var(--text-color);
    opacity: 0.60;
    font-size: 0.82rem;
    margin-top: 4px;
}

[data-testid="stMetric"] {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 16px;
    padding: 16px 17px;
    min-height: 108px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
}

[data-testid="stMetricLabel"] {
    color: var(--text-color) !important;
    opacity: 0.65;
    font-size: 0.76rem;
    font-weight: 650;
}

[data-testid="stMetricValue"] {
    color: var(--text-color) !important;
    font-size: 1.62rem;
    font-weight: 760;
}

[data-testid="stMetricDelta"] {
    color: var(--text-color) !important;
    opacity: 0.65;
}

.insight-card {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 16px;
    padding: 18px 19px;
    min-height: 132px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
}

.insight-label {
    color: var(--text-color);
    opacity: 0.60;
    font-size: 0.72rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.insight-value {
    color: var(--text-color);
    font-size: 1.35rem;
    font-weight: 750;
    margin-top: 7px;
}

.insight-copy {
    color: var(--text-color);
    opacity: 0.65;
    font-size: 0.8rem;
    line-height: 1.5;
    margin-top: 5px;
}

.recommendation {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 16px;
    padding: 18px 20px;
    color: var(--text-color);
    font-size: 0.9rem;
    line-height: 1.6;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
}

.priority {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 13px;
    padding: 15px 16px;
    color: var(--text-color);
    font-size: 0.9rem;
    line-height: 1.5;
}

.priority strong {
    display: block;
    color: var(--text-color);
    font-size: 0.76rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.priority-critical {
    border-color: rgba(217,75,75,0.35);
    background-color: rgba(217,75,75,0.10);
}

.priority-high {
    border-color: rgba(201,133,18,0.35);
    background-color: rgba(201,133,18,0.10);
}

.priority-medium {
    border-color: rgba(47,125,246,0.30);
    background-color: rgba(47,125,246,0.10);
}

.priority-low {
    border-color: rgba(24,167,102,0.30);
    background-color: rgba(24,167,102,0.10);
}

.arch-card {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 16px;
    padding: 19px;
    min-height: 170px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
}

.arch-index {
    width: 31px;
    height: 31px;
    display: grid;
    place-items: center;
    border-radius: 9px;
    background: rgba(47,125,246,0.10);
    border: 1px solid rgba(47,125,246,0.20);
    color: #2f7df6;
    font-size: 0.76rem;
    font-weight: 800;
}

.arch-title {
    color: var(--text-color);
    font-size: 1rem;
    font-weight: 730;
    margin-top: 13px;
}

.arch-copy {
    color: var(--text-color);
    opacity: 0.65;
    font-size: 0.8rem;
    line-height: 1.55;
    margin-top: 6px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.25);
    background-color: var(--secondary-background-color);
    color: var(--text-color);
    min-height: 42px;
    font-weight: 650;
}

.stButton > button:hover {
    border-color: rgba(47,125,246,0.45);
    background-color: var(--background-color);
    color: var(--text-color);
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    background-color: var(--secondary-background-color);
    color: var(--text-color);
}

.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stMultiSelect label,
.stSlider label,
.stFileUploader label {
    color: var(--text-color) !important;
}

div[data-testid="stDataFrame"] {
    background-color: var(--secondary-background-color);
    border-radius: 12px;
}

div[data-testid="stAlert"] {
    border-radius: 13px;
}

hr {
    border-color: rgba(128,128,128,0.25) !important;
}

.footer {
    text-align: center;
    color: var(--text-color);
    opacity: 0.55;
    font-size: 0.74rem;
    padding-top: 15px;
}

@media (max-width: 900px) {

    .block-container {
        padding: 1.2rem 1rem 2rem;
    }

    .hero-title {
        font-size: 1.9rem;
    }

}

</style>
""", unsafe_allow_html=True)


# Load model and dataset

@st.cache_resource
def load_model():
    model = joblib.load("conveyor_model.pkl")
    encoder = joblib.load("fault_encoder.pkl")
    return model, encoder


@st.cache_data
def load_dataset():
    return pd.read_csv("conveyor_data.csv")


model, encoder = load_model()
df = load_dataset()


# Sensor features

features = [
    "vibration",
    "temperature",
    "belt_speed",
    "tension",
    "motor_current",
    "alignment"
]


# Health score

def calculate_health_score(fault):

    scores = {
        "Normal": 95,
        "Misalignment": 70,
        "Overload": 60,
        "Belt Damage": 40,
        "Joint Failure": 20
    }

    return scores.get(fault, 50)


# Maintenance recommendation

def maintenance_recommendation(fault):

    recommendations = {

        "Normal":
        "No immediate maintenance required. Continue normal monitoring.",

        "Misalignment":
        "Inspect belt alignment, rollers and tracking system.",

        "Overload":
        "Check material loading and motor loading conditions.",

        "Belt Damage":
        "Inspect belt surface and damaged sections.",

        "Joint Failure":
        "Inspect belt joint immediately and schedule maintenance."
    }

    return recommendations.get(
        fault,
        "Perform detailed conveyor inspection."
    )


# Prediction function

def predict_condition(sensor_data):

    input_data = pd.DataFrame([sensor_data])

    prediction = model.predict(input_data)

    fault = encoder.inverse_transform(prediction)[0]

    probabilities = model.predict_proba(input_data)[0]

    confidence = max(probabilities) * 100

    health = calculate_health_score(fault)

    recommendation = maintenance_recommendation(fault)

    return fault, confidence, health, recommendation, probabilities


# Sidebar

with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">
            Conveyor Health Monitor
        </div>
        <div class="sidebar-brand-subtitle">
            AI-powered predictive maintenance
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### System Status")

    st.markdown("""
    <div class="status-card">
        <span class="status-dot"></span>
        AI Model Online
    </div>

    <div class="status-card">
        <span class="status-dot"></span>
        Sensor Feed Active
    </div>

    <div class="status-card">
        <span class="status-dot"></span>
        Monitoring Active
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Prediction Mode")

    mode = st.radio(
        "Select mode",
        ["Manual Sensor Input", "Dataset Sample"],
        label_visibility="collapsed"
    )


# Header

st.markdown("""
<div class="hero">

<div class="hero-kicker">
AI Conveyor Monitoring System
</div>

<div class="hero-title">
Iron Ore Conveyor Health Monitor
</div>

<div class="hero-subtitle">
Intelligent condition monitoring and predictive maintenance
for conveyor belts using machine learning and sensor data.
</div>

<div class="hero-badge">
<span class="status-dot"></span>
AI Monitoring Active
</div>

</div>
""", unsafe_allow_html=True)


# Manual sensor input

if mode == "Manual Sensor Input":

    st.markdown(
        '<div class="section-title">Live Sensor Input</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Enter simulated sensor readings</div>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:

        vibration = st.number_input(
            "Vibration",
            min_value=0.5,
            max_value=15.0,
            value=3.0,
            step=0.1
        )

        temperature = st.number_input(
            "Temperature",
            min_value=30.0,
            max_value=100.0,
            value=55.0,
            step=0.5
        )

    with col2:

        belt_speed = st.number_input(
            "Belt Speed",
            min_value=1.5,
            max_value=3.5,
            value=2.8,
            step=0.1
        )

        tension = st.number_input(
            "Belt Tension",
            min_value=45.0,
            max_value=160.0,
            value=95.0,
            step=1.0
        )

    with col3:

        motor_current = st.number_input(
            "Motor Current",
            min_value=50.0,
            max_value=180.0,
            value=105.0,
            step=1.0
        )

        alignment = st.number_input(
            "Alignment Deviation",
            min_value=0.05,
            max_value=6.0,
            value=0.8,
            step=0.1
        )

    sensor_data = {
        "vibration": vibration,
        "temperature": temperature,
        "belt_speed": belt_speed,
        "tension": tension,
        "motor_current": motor_current,
        "alignment": alignment
    }


# Dataset sample mode

else:

    st.markdown(
        '<div class="section-title">Dataset Simulation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Use an existing simulated sensor record</div>',
        unsafe_allow_html=True
    )

    sample_index = st.slider(
        "Select dataset sample",
        0,
        len(df) - 1,
        0
    )

    sample = df.iloc[sample_index]

    sensor_data = {
        "vibration": sample["vibration"],
        "temperature": sample["temperature"],
        "belt_speed": sample["belt_speed"],
        "tension": sample["tension"],
        "motor_current": sample["motor_current"],
        "alignment": sample["alignment"]
    }


# Prediction

fault, confidence, health, recommendation, probabilities = predict_condition(
    sensor_data
)


# Main metrics

st.write("")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Conveyor Health",
        f"{health}/100"
    )

with col2:
    st.metric(
        "Detected Condition",
        fault
    )

with col3:
    st.metric(
        "AI Confidence",
        f"{confidence:.1f}%"
    )

with col4:
    st.metric(
        "Operating Status",
        "Healthy" if health >= 80 else "Attention Required"
    )


# Maintenance recommendation

st.write("")

st.markdown(
    '<div class="section-title">Maintenance Recommendation</div>',
    unsafe_allow_html=True
)

st.write("")

if health >= 80:
    priority_class = "priority-low"
    priority_text = "LOW PRIORITY"
elif health >= 60:
    priority_class = "priority-medium"
    priority_text = "MEDIUM PRIORITY"
elif health >= 40:
    priority_class = "priority-high"
    priority_text = "HIGH PRIORITY"
else:
    priority_class = "priority-critical"
    priority_text = "CRITICAL"


st.markdown(
    f"""
    <div class="priority {priority_class}">
        <strong>{priority_text}</strong>
        {recommendation}
    </div>
    """,
    unsafe_allow_html=True
)


# Sensor overview

st.write("")

st.markdown(
    '<div class="section-title">Current Sensor Readings</div>',
    unsafe_allow_html=True
)

st.write("")

sensor_col1, sensor_col2, sensor_col3 = st.columns(3)

with sensor_col1:

    st.metric(
        "Vibration",
        f"{sensor_data['vibration']:.2f}"
    )

    st.metric(
        "Temperature",
        f"{sensor_data['temperature']:.1f} °C"
    )

with sensor_col2:

    st.metric(
        "Belt Speed",
        f"{sensor_data['belt_speed']:.2f}"
    )

    st.metric(
        "Belt Tension",
        f"{sensor_data['tension']:.1f}"
    )

with sensor_col3:

    st.metric(
        "Motor Current",
        f"{sensor_data['motor_current']:.1f} A"
    )

    st.metric(
        "Alignment",
        f"{sensor_data['alignment']:.2f}"
    )


# Fault probability

st.write("")

st.markdown(
    '<div class="section-title">AI Fault Probability</div>',
    unsafe_allow_html=True
)

probability_df = pd.DataFrame({
    "Fault": encoder.classes_,
    "Probability": probabilities * 100
})

probability_df = probability_df.sort_values(
    "Probability",
    ascending=False
)

st.bar_chart(
    probability_df.set_index("Fault")
)


# Sensor trends

st.write("")

st.markdown(
    '<div class="section-title">Sensor Trends</div>',
    unsafe_allow_html=True
)

trend_df = df[features].head(100).copy()

st.line_chart(trend_df)


# Feature importance

st.write("")

st.markdown(
    '<div class="section-title">AI Feature Importance</div>',
    unsafe_allow_html=True
)

importance_df = pd.DataFrame({
    "Sensor": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("Sensor")
)


# Dataset preview

st.write("")

st.markdown(
    '<div class="section-title">Dataset Preview</div>',
    unsafe_allow_html=True
)

st.dataframe(
    df.head(10),
    use_container_width=True
)


# System architecture

st.write("")

st.markdown(
    '<div class="section-title">System Architecture</div>',
    unsafe_allow_html=True
)

st.write("")

arch1, arch2, arch3 = st.columns(3)

with arch1:

    st.markdown("""
    <div class="arch-card">

    <div class="arch-index">01</div>

    <div class="arch-title">
    Sensor Layer
    </div>

    <div class="arch-copy">
    Vibration, temperature, speed, tension,
    motor current and alignment measurements
    are collected from the conveyor.
    </div>

    </div>
    """, unsafe_allow_html=True)


with arch2:

    st.markdown("""
    <div class="arch-card">

    <div class="arch-index">02</div>

    <div class="arch-title">
    AI Prediction
    </div>

    <div class="arch-copy">
    The machine-learning model analyzes
    sensor patterns and identifies abnormal
    conveyor operating conditions.
    </div>

    </div>
    """, unsafe_allow_html=True)


with arch3:

    st.markdown("""
    <div class="arch-card">

    <div class="arch-index">03</div>

    <div class="arch-title">
    Maintenance Decision
    </div>

    <div class="arch-copy">
    The system generates a health score,
    fault classification and maintenance
    recommendation.
    </div>

    </div>
    """, unsafe_allow_html=True)


# Footer

st.markdown("""
<div class="footer">
SIH26008 • Smart Automation • AI-Based Conveyor Health Monitoring
</div>
""", unsafe_allow_html=True)
