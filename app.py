"""
Daily Minimum Temperature Forecaster - Melbourne, Australia

"""

import base64
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import xgboost as xgb

st.set_page_config(
    page_title="Melbourne Min Temp Forecaster",
    page_icon="🌡️",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Background artwork: a decorative image sits behind everything on the full
# page, but all real content (inputs, metrics, tables, chart) lives inside an
# opaque panel on top of it -- so the art is visible only in the margins and
# never reduces legibility of anything the user needs to read or click.
# ---------------------------------------------------------------------------
def inject_background():
    try:
        with open("assets/bg_original.svg", "r") as f:
            svg_bytes = f.read().encode("utf-8")
        b64 = base64.b64encode(svg_bytes).decode("ascii")
    except FileNotFoundError:
        return  # app still works fine with no decorative background

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/svg+xml;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        .block-container {{
            background-color: rgba(14, 17, 23, 0.90);
            border-radius: 16px;
            padding: 2rem 2.5rem 3rem 2.5rem;
            margin-top: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_background()

# ---------------------------------------------------------------------------
# Load model + data (cached so this only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    lr_model = joblib.load("lr_model.pkl")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("xgb_model.json")
    meta = joblib.load("model_meta.pkl")
    return lr_model, xgb_model, meta


@st.cache_data
def load_history():
    hist = pd.read_csv("temperature_history.csv")
    hist["Date"] = pd.to_datetime(hist["Date"])
    return hist


lr_model, xgb_model, meta = load_models()
history = load_history()
FEATURES = meta["features"]

MIN_DATE = pd.Timestamp(meta["min_predict_date"]).date()   # earliest date with full lag_365 history
MAX_DATE = pd.Timestamp(meta["data_end"]).date()            # last date in the historical record


# ---------------------------------------------------------------------------
# Feature engineering - mirrors the notebook exactly
# ---------------------------------------------------------------------------
def compute_features_for_date(hist_df: pd.DataFrame, target_date: pd.Timestamp):
    """Compute the 11 engineered features needed to predict target_date's
    minimum temperature, using only data STRICTLY BEFORE target_date
    (i.e. what would genuinely be known the night before)."""
    prior = hist_df[hist_df["Date"] < target_date].sort_values("Date")
    if len(prior) < 365:
        return None, None

    temp_series = prior.set_index("Date")["Temp"]
    doy = target_date.dayofyear

    feats = {
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "lag_1": temp_series.iloc[-1],
        "lag_2": temp_series.iloc[-2],
        "lag_3": temp_series.iloc[-3],
        "lag_7": temp_series.iloc[-7],
        "lag_14": temp_series.iloc[-14],
        "lag_365": temp_series.iloc[-365],
        "roll_mean_7": temp_series.iloc[-7:].mean(),
        "roll_std_7": temp_series.iloc[-7:].std(),
        "roll_mean_30": temp_series.iloc[-30:].mean(),
    }
    recent_30 = prior.tail(30)
    return feats, recent_30


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🌡️ Melbourne Minimum Temperature Forecaster")
st.caption("Capstone 2 · Predicts tomorrow's minimum temperature from historical readings")

st.markdown("### Step 1 - Choose a date")
st.write(
    "Pick a date and the app will forecast the **minimum temperature for the "
    "following day**, using only the history strictly before it - exactly the "
    "information that would genuinely be available the night before."
)

selected_date = st.date_input(
    "Forecast base date",
    value=pd.Timestamp("1990-06-15").date(),
    min_value=MIN_DATE,
    max_value=MAX_DATE,
    help="The app forecasts the day AFTER this date.",
)
target_date = pd.Timestamp(selected_date) + pd.Timedelta(days=1)

model_choice = st.selectbox(
    "Model",
    options=["Linear Regression (recommended)", "Blend (Linear Regression + XGBoost)", "XGBoost (tuned)"],
    index=0,
    help="See the Model Development Document, Section 8.1, for why Linear Regression is the recommended deployment model.",
)

st.markdown("### Step 2 - Run the forecast")
run = st.button("Predict", type="primary", use_container_width=True)

if run:
    feats, recent_30 = compute_features_for_date(history, target_date)

    if feats is None:
        st.error(
            f"Not enough trailing history before {target_date.date()} to compute "
            f"all features (lag_365 needs a full year of prior data). "
            f"Please choose a date on or after {MIN_DATE}."
        )
    else:
        X_input = pd.DataFrame([feats])[FEATURES]

        pred_lr = float(lr_model.predict(X_input)[0])
        pred_xgb = float(xgb_model.predict(X_input)[0])
        pred_blend = 0.6 * pred_lr + 0.4 * pred_xgb

        if model_choice.startswith("Linear"):
            prediction, resid_std, model_label = pred_lr, meta["resid_std_lr"], "Linear Regression"
        elif model_choice.startswith("Blend"):
            prediction, resid_std, model_label = pred_blend, meta["resid_std_blend"], "Blend (LR + XGBoost)"
        else:
            prediction, resid_std, model_label = pred_xgb, meta["resid_std_lr"], "XGBoost (tuned)"

        st.markdown("### Result")
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Forecast for {target_date.date()}", f"{prediction:.1f} °C")
        col2.metric("Typical range (±1 std)", f"{prediction - resid_std:.1f} to {prediction + resid_std:.1f} °C")

        actual_row = history[history["Date"] == target_date]
        if not actual_row.empty:
            actual = float(actual_row["Temp"].iloc[0])
            col3.metric("Actual (historical record)", f"{actual:.1f} °C", delta=f"{prediction - actual:+.1f} °C")
        else:
            col3.metric("Actual", "Not in dataset")

        naive_pred = feats["lag_1"]
        st.caption(
            f"For comparison, the naive persistence baseline (\"tomorrow = today\") "
            f"would have forecast **{naive_pred:.1f} °C**."
        )

        # --- Chart: trailing 30 days + forecast point ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_30["Date"], y=recent_30["Temp"],
            mode="lines+markers", name="Actual (last 30 days)",
            line=dict(color="#4FD1C5"),
        ))
        fig.add_trace(go.Scatter(
            x=[target_date], y=[prediction],
            mode="markers", name=f"Forecast ({model_label})",
            marker=dict(color="#F56565", size=12, symbol="star"),
        ))
        if not actual_row.empty:
            fig.add_trace(go.Scatter(
                x=[target_date], y=[actual],
                mode="markers", name="Actual (if known)",
                marker=dict(color="#68D391", size=10),
            ))
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=30, b=10),
            height=380,
            xaxis_title="Date", yaxis_title="Min Temp (°C)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("See the auto-computed engineered features"):
            feat_df = pd.DataFrame([feats]).T.rename(columns={0: "Value"})
            feat_df["Value"] = feat_df["Value"].round(3)
            st.dataframe(feat_df, use_container_width=True)

else:
    st.info("Choose a date and model above, then click **Predict**.")

st.divider()
