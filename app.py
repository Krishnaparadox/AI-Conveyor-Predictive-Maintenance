import streamlit as st
import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Conveyor AI Predictive Maintenance",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dashboard theme

st.markdown(
    """
    <style>
        /* Theme-aware color system */
        :root {
            --app-bg: var(--background-color, #f6f8fb);
            --surface: var(--secondary-background-color, #ffffff);
            --surface-soft: color-mix(in srgb, var(--surface) 88%, var(--app-bg));
            --text: var(--text-color, #172033);
            --muted: color-mix(in srgb, var(--text) 58%, transparent);
            --border: color-mix(in srgb, var(--text) 12%, transparent);
            --accent: #2f7df6;
            --accent-soft: rgba(47, 125, 246, 0.10);
            --green: #18a766;
            --amber: #c98512;
            --red: #d94b4b;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                         BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 100% 0%, rgba(47,125,246,0.055), transparent 24%),
                var(--app-bg);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stAppViewContainer"] main {
            color: var(--text);
        }

        [data-testid="stAppViewContainer"] main h1,
        [data-testid="stAppViewContainer"] main h2,
        [data-testid="stAppViewContainer"] main h3,
        [data-testid="stAppViewContainer"] main h4,
        [data-testid="stAppViewContainer"] main p,
        [data-testid="stAppViewContainer"] main label {
            color: var(--text);
        }

        .block-container {
            max-width: 1480px;
            padding: 1.8rem 2.25rem 3.1rem;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: var(--secondary-background-color, #ffffff);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding: 1.45rem 1rem;
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
        }

        [data-testid="stSidebar"] label {
            color: var(--muted) !important;
            font-size: 0.8rem;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: var(--app-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
        }

        .sidebar-brand {
            margin-bottom: 1.35rem;
        }

        .sidebar-brand-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 750;
            letter-spacing: -0.02em;
        }

        .sidebar-brand-subtitle {
            color: var(--muted);
            font-size: 0.73rem;
            margin-top: 3px;
        }

        .status-card {
            display: flex;
            align-items: center;
            gap: 9px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 11px;
            margin: 7px 0;
            font-size: 0.81rem;
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 0 4px rgba(24,167,102,0.10);
            flex: 0 0 auto;
        }

        /* Hero */
        .hero {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(47,125,246,0.10), transparent 58%),
                var(--surface);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 30px 34px 28px;
            margin-bottom: 24px;
            box-shadow: 0 14px 40px rgba(15,23,42,0.07);
        }

        .hero::after {
            content: "";
            position: absolute;
            right: -80px;
            top: -100px;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            background: rgba(47,125,246,0.06);
            filter: blur(8px);
        }

        .hero-kicker {
            color: var(--accent);
            font-size: 0.73rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 8px;
        }

        .hero-title {
            color: var(--text);
            font-size: 2.45rem;
            line-height: 1.08;
            font-weight: 780;
            letter-spacing: -0.045em;
            margin: 0;
        }

        .hero-subtitle {
            color: var(--muted);
            max-width: 830px;
            font-size: 0.96rem;
            line-height: 1.6;
            margin-top: 10px;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            margin-top: 17px;
            background: rgba(24,167,102,0.08);
            border: 1px solid rgba(24,167,102,0.20);
            color: var(--green);
            border-radius: 999px;
            padding: 7px 11px;
            font-size: 0.76rem;
            font-weight: 700;
        }

        /* Section headers */
        .section-title {
            color: var(--text);
            font-size: 1.16rem;
            font-weight: 730;
            letter-spacing: -0.02em;
            margin: 0;
        }

        .section-subtitle {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 4px;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px 17px;
            min-height: 108px;
            box-shadow: 0 8px 24px rgba(15,23,42,0.045);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted) !important;
            font-size: 0.76rem;
            font-weight: 650;
        }

        [data-testid="stMetricValue"] {
            color: var(--text) !important;
            font-size: 1.62rem;
            font-weight: 760;
            letter-spacing: -0.025em;
        }

        [data-testid="stMetricDelta"] {
            color: var(--muted) !important;
        }

        /* Info and recommendation cards */
        .insight-card,
        .recommendation,
        .arch-card,
        .priority {
            background: var(--surface);
            border: 1px solid var(--border);
            box-shadow: 0 8px 24px rgba(15,23,42,0.045);
        }

        .insight-card {
            border-radius: 16px;
            padding: 18px 19px;
            min-height: 132px;
        }

        .insight-label {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .insight-value {
            color: var(--text);
            font-size: 1.35rem;
            font-weight: 750;
            margin-top: 7px;
        }

        .insight-copy {
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.5;
            margin-top: 5px;
        }

        .priority {
            border-radius: 13px;
            padding: 15px 16px;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .priority strong {
            display: block;
            color: var(--text);
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }

        .priority-critical {
            border-color: rgba(217,75,75,0.28);
            background: rgba(217,75,75,0.055);
        }

        .priority-high {
            border-color: rgba(201,133,18,0.28);
            background: rgba(201,133,18,0.055);
        }

        .priority-medium {
            border-color: rgba(47,125,246,0.24);
            background: rgba(47,125,246,0.055);
        }

        .priority-low {
            border-color: rgba(24,167,102,0.24);
            background: rgba(24,167,102,0.055);
        }

        .recommendation {
            border-radius: 16px;
            padding: 18px 20px;
            color: var(--text);
            font-size: 0.9rem;
            line-height: 1.6;
        }

        /* Charts: softer containers and spacing */
        div[data-testid="stVegaLiteChart"] {
            background: var(--surface) !important;
            border: 1px solid var(--border);
            border-radius: 18px;
            overflow: hidden;
            padding: 10px 10px 4px;
            box-shadow: 0 8px 24px rgba(15,23,42,0.045);
        }

        /* Native Streamlit messages */
        div[data-testid="stAlert"] {
            border-radius: 13px;
            border-width: 1px;
        }

        /* Architecture cards */
        .arch-card {
            border-radius: 16px;
            padding: 19px;
            min-height: 170px;
        }

        .arch-index {
            width: 31px;
            height: 31px;
            display: grid;
            place-items: center;
            border-radius: 9px;
            background: var(--accent-soft);
            border: 1px solid rgba(47,125,246,0.18);
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 800;
        }

        .arch-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 730;
            margin-top: 13px;
        }

        .arch-copy {
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.55;
            margin-top: 6px;
        }

        .footer {
            text-align: center;
            color: var(--muted);
            font-size: 0.74rem;
            padding-top: 15px;
        }

        /* Buttons */
        .stButton > button {
            width: 100%;
            border-radius: 10px;
            border: 1px solid var(--border);
            background: var(--surface);
            color: var(--text);
            min-height: 42px;
            font-weight: 650;
        }

        .stButton > button:hover {
            border-color: rgba(47,125,246,0.40);
            background: var(--surface-soft);
            color: var(--text);
        }

        hr {
            border-color: var(--border) !important;
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
    """,
    unsafe_allow_html=True,
)

# Model and dataset

# Refresh the simulated sensor readings every 3 seconds
st_autorefresh(interval=3000, key="sensor_refresh")

# Load the trained model and label encoder
model = joblib.load("model/conveyor_model.pkl")
encoder = joblib.load("model/fault_encoder.pkl")

data_path = "data/conveyor_data.csv"
dataset = pd.read_csv(data_path)

# Sensor feature names
feature_names = [
    "Vibration",
    "Temperature",
    "Belt Speed",
    "Tension",
    "Motor Current",
    "Alignment",
]

# Available operating scenarios
fault_conditions = [
    "Normal",
    "Misalignment",
    "Overload",
    "Belt Damage",
    "Joint Failure",
]

# Business logic

def calculate_health_score(fault):
    scores = {
        "Normal": 95,
        "Misalignment": 70,
        "Overload": 60,
        "Belt Damage": 40,
        "Joint Failure": 20,
    }
    return scores.get(fault, 50)

def maintenance_recommendation(fault):
    recommendations = {
        "Normal": "No immediate maintenance required. Continue monitoring.",
        "Misalignment": "Inspect belt alignment, rollers and tracking system.",
        "Overload": "Check material load and motor loading conditions.",
        "Belt Damage": "Inspect belt surface and damaged sections.",
        "Joint Failure": "Immediately inspect belt joint and schedule maintenance.",
    }
    return recommendations.get(fault, "Perform detailed conveyor inspection.")

def maintenance_priority(fault):
    priorities = {
        "Normal": "LOW",
        "Misalignment": "MEDIUM",
        "Overload": "HIGH",
        "Belt Damage": "HIGH",
        "Joint Failure": "CRITICAL",
    }
    return priorities.get(fault, "MEDIUM")

def failure_risk_score(fault, confidence):
    base_scores = {
        "Normal": 10,
        "Misalignment": 35,
        "Overload": 60,
        "Belt Damage": 75,
        "Joint Failure": 90,
    }

    base_score = base_scores.get(fault, 50)
    confidence_factor = confidence / 100
    return round(base_score * confidence_factor)

def early_warning(risk_score):
    if risk_score >= 80:
        return "CRITICAL"
    if risk_score >= 60:
        return "HIGH"
    if risk_score >= 35:
        return "MEDIUM"
    return "LOW"

def calculate_degradation(history_df):
    if len(history_df) < 5:
        return 0

    recent = history_df.tail(5)
    older = history_df.head(5)

    vibration_change = max(
        0,
        recent["Vibration"].mean() - older["Vibration"].mean(),
    )

    temperature_change = max(
        0,
        recent["Temperature"].mean() - older["Temperature"].mean(),
    )

    current_change = max(
        0,
        recent["Motor Current"].mean() - older["Motor Current"].mean(),
    )

    score = (
        vibration_change * 8
        + temperature_change * 2
        + current_change * 0.5
    )

    return round(min(score, 100))

def priority_class(value):
    return value.lower()

# Header

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">INDUSTRIAL CONDITION MONITORING</div>
        <div class="hero-title">Conveyor AI Predictive Maintenance</div>
        <div class="hero-subtitle">
            Multi-sensor monitoring, machine-learning fault classification
            and maintenance decision support in a single operational dashboard.
        </div>
        <div class="hero-badge">
            <span class="status-dot"></span>
            AI model online
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">Conveyor Monitoring</div>
        <div class="sidebar-brand-subtitle">AI maintenance control panel</div>
    </div>
    """,
    unsafe_allow_html=True,
)

scenario = st.sidebar.selectbox(
    "Operating Scenario",
    fault_conditions,
)

if st.sidebar.button("Reset Sensor History"):
    st.session_state.sensor_history = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")

st.sidebar.markdown(
    '<div class="status-card"><span class="status-dot"></span>AI model online</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="status-card"><span class="status-dot"></span>Sensor simulation active</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="status-card">Refresh interval: 3 seconds</div>',
    unsafe_allow_html=True,
)

# Session state

if "sensor_history" not in st.session_state:
    st.session_state.sensor_history = []

# Sensor reading

scenario_data = dataset[dataset["fault_condition"] == scenario]

if not scenario_data.empty:
    selected_row = scenario_data.sample(n=1).iloc[0]

    vibration = selected_row["vibration"]
    temperature = selected_row["temperature"]
    belt_speed = selected_row["belt_speed"]
    tension = selected_row["tension"]
    motor_current = selected_row["motor_current"]
    alignment = selected_row["alignment"]

    st.session_state.sensor_history.append(
        {
            "Vibration": vibration,
            "Temperature": temperature,
            "Belt Speed": belt_speed,
            "Tension": tension,
            "Motor Current": motor_current,
            "Alignment": alignment,
        }
    )

    if len(st.session_state.sensor_history) > 100:
        st.session_state.sensor_history = (
            st.session_state.sensor_history[-100:]
        )

    # Live sensor monitoring

    st.markdown(
        """
        <div class="section">
            <div class="section-title">Live Sensor Monitoring</div>
            <div class="section-subtitle">
                Current operating values from the active conveyor scenario
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sensor_cols = st.columns(6)

    sensor_values = [
        ("Vibration", f"{vibration:.2f}"),
        ("Temperature", f"{temperature:.2f}"),
        ("Belt Speed", f"{belt_speed:.2f}"),
        ("Tension", f"{tension:.2f}"),
        ("Motor Current", f"{motor_current:.2f}"),
        ("Alignment", f"{alignment:.2f}"),
    ]

    for column, (label, value) in zip(sensor_cols, sensor_values):
        with column:
            st.metric(label, value)

    # AI prediction

    sensor_data = {
        "vibration": vibration,
        "temperature": temperature,
        "belt_speed": belt_speed,
        "tension": tension,
        "motor_current": motor_current,
        "alignment": alignment,
    }

    data = pd.DataFrame([sensor_data])

    prediction = model.predict(data)
    fault = encoder.inverse_transform(prediction)[0]

    probabilities = model.predict_proba(data)[0]

    probability_map = dict(
        zip(encoder.classes_, probabilities)
    )

    confidence = max(probabilities) * 100
    health_score = calculate_health_score(fault)
    risk_score = failure_risk_score(fault, confidence)
    warning_level = early_warning(risk_score)
    priority = maintenance_priority(fault)
    recommendation = maintenance_recommendation(fault)

    # AI condition analysis

    st.markdown(
        """
        <div class="section">
            <div class="section-title">AI Condition Analysis</div>
            <div class="section-subtitle">
                Machine-learning assessment of the current conveyor state
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    analysis_cols = st.columns(4)

    analysis_values = [
        ("Detected Condition", fault),
        ("AI Confidence", f"{confidence:.1f}%"),
        ("Conveyor Health", f"{health_score}/100"),
        ("Failure Risk", f"{risk_score}/100"),
    ]

    for column, (label, value) in zip(analysis_cols, analysis_values):
        with column:
            st.metric(label, value)

    # Maintenance status

    status_cols = st.columns(2)

    with status_cols[0]:
        st.markdown(
            '<div class="section-title">Maintenance Priority</div>'
            '<div class="section-subtitle">Recommended response level</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="priority priority-{priority_class(priority)}">
                <strong>{priority}</strong>
                {(
                    "Immediate maintenance is required."
                    if priority == "CRITICAL"
                    else "Maintenance should be scheduled soon."
                    if priority == "HIGH"
                    else "Inspect the conveyor system and continue monitoring."
                    if priority == "MEDIUM"
                    else "System operating within the normal condition range."
                )}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with status_cols[1]:
        st.markdown(
            '<div class="section-title">Early Warning Level</div>'
            '<div class="section-subtitle">Current risk classification</div>',
            unsafe_allow_html=True,
        )

        warning_text = {
            "CRITICAL": "Immediate attention recommended.",
            "HIGH": "Elevated failure risk detected.",
            "MEDIUM": "Moderate risk; monitor operating trends.",
            "LOW": "Low current failure risk.",
        }[warning_level]

        st.markdown(
            f"""
            <div class="priority priority-{priority_class(warning_level)}">
                <strong>{warning_level}</strong>
                {warning_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Degradation monitor

    st.markdown(
        """
        <div class="section">
            <div class="section-title">Conveyor Degradation Monitor</div>
            <div class="section-subtitle">
                Compares recent sensor behaviour with earlier readings
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history_df = pd.DataFrame(st.session_state.sensor_history)
    degradation_score = calculate_degradation(history_df)

    if degradation_score >= 70:
        degradation_status = "CRITICAL"
    elif degradation_score >= 45:
        degradation_status = "WARNING"
    elif degradation_score >= 20:
        degradation_status = "WATCH"
    else:
        degradation_status = "STABLE"

    degradation_cols = st.columns(3)

    with degradation_cols[0]:
        st.metric("Degradation Score", f"{degradation_score}/100")

    with degradation_cols[1]:
        st.metric("Trend Status", degradation_status)

    with degradation_cols[2]:
        st.metric("Samples Collected", len(history_df))

    if len(history_df) < 5:
        st.info(
            "Collecting sensor history. The degradation monitor becomes "
            "more informative after several readings."
        )
    elif degradation_score >= 70:
        st.error(
            "The recent sensor trend indicates significant degradation. "
            "A maintenance inspection is recommended."
        )
    elif degradation_score >= 45:
        st.warning(
            "The recent sensor trend indicates increasing operating stress. "
            "Continue monitoring and prepare maintenance."
        )
    else:
        st.success("Sensor trends are currently stable.")

    # Recommendation

    st.markdown(
        """
        <div class="section">
            <div class="section-title">AI Maintenance Recommendation</div>
            <div class="section-subtitle">
                Decision support generated from the detected condition
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="recommendation">{recommendation}</div>',
        unsafe_allow_html=True,
    )

    # Fault probability

    st.markdown(
        """
        <div class="section">
            <div class="section-title">Fault Probability</div>
            <div class="section-subtitle">
                Model probability distribution across known conveyor conditions
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    probability_df = pd.DataFrame(
        {
            "Condition": list(probability_map.keys()),
            "Probability": [
                probability_map[condition] * 100
                for condition in probability_map.keys()
            ],
        }
    ).sort_values("Probability", ascending=False)

    st.bar_chart(
        probability_df.set_index("Condition"),
        height=310,
    )

    # Feature importance

    st.markdown(
        """
        <div class="section">
            <div class="section-title">Model Feature Importance</div>
            <div class="section-subtitle">
                Relative contribution of each sensor feature to classification
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    importance_df = pd.DataFrame(
        {
            "Sensor": feature_names,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    st.bar_chart(
        importance_df.set_index("Sensor"),
        height=310,
    )

    st.caption(
        "Higher importance indicates that the model relies more heavily on "
        "that sensor when identifying conveyor conditions."
    )

# Live sensor trends

if st.session_state.sensor_history:
    st.markdown(
        """
        <div class="section">
            <div class="section-title">Live Sensor Trends</div>
            <div class="section-subtitle">
                Recent sensor history collected during the active session
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history_df = pd.DataFrame(st.session_state.sensor_history)

    chart_cols = st.columns(2)

    charts = [
        ("Vibration", "Vibration"),
        ("Belt Speed", "Belt Speed"),
        ("Motor Current", "Motor Current"),
        ("Temperature", "Temperature"),
        ("Tension", "Tension"),
        ("Alignment", "Alignment"),
    ]

    for index, (title, column_name) in enumerate(charts):
        with chart_cols[index % 2]:
            st.markdown(
                f'<div class="chart-title">{title}</div>',
                unsafe_allow_html=True,
            )
            st.line_chart(
                history_df[[column_name]],
                height=245,
            )

# System architecture

st.markdown(
    """
    <div class="section">
        <div class="section-title">System Architecture</div>
        <div class="section-subtitle">
            From sensor measurements to maintenance decisions
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

architecture = [
    (
        "01",
        "Sensors",
        "Collect vibration, temperature, belt speed, tension, motor current and alignment data.",
    ),
    (
        "02",
        "AI Model",
        "Random Forest analyzes the sensor pattern and classifies the conveyor condition.",
    ),
    (
        "03",
        "Risk Engine",
        "The model output is translated into health, risk and degradation indicators.",
    ),
    (
        "04",
        "Maintenance",
        "The platform converts the assessment into a practical maintenance action.",
    ),
]

arch_cols = st.columns(4)

for column, (number, title, copy) in zip(arch_cols, architecture):
    with column:
        st.markdown(
            f"""
            <div class="arch-card">
                <div class="arch-index">{number}</div>
                <div class="arch-title">{title}</div>
                <div class="arch-copy">{copy}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="footer">
        SIH Prototype · AI-based Conveyor Condition Monitoring ·
        Predictive Maintenance Decision Support
    </div>
    """,
    unsafe_allow_html=True,
)
