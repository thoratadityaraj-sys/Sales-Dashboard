import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Anomaly Report", page_icon="🚨", layout="wide")

st.title("🚨 Anomaly Report")
st.caption("Page 3 of 4 — Weekly sales anomalies detected via Isolation Forest and a rolling Z-score "
           "method (Task 5).")


@st.cache_data
def load_anomalies():
    a = pd.read_csv("data/anomalies.csv")
    a["Week"] = pd.to_datetime(a["Week"])
    return a


weekly = load_anomalies()

method = st.radio(
    "Detection method to display:",
    ["Isolation Forest", "Z-Score (rolling)", "Both (agreement view)"],
    horizontal=True
)

st.divider()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=weekly["Week"], y=weekly["Weekly Sales"], mode="lines",
    name="Weekly Sales", line=dict(color="lightgray", width=1.5)
))

if method == "Isolation Forest":
    anomalies = weekly[weekly["iso_anomaly"]]
    fig.add_trace(go.Scatter(
        x=anomalies["Week"], y=anomalies["Weekly Sales"], mode="markers",
        name="Isolation Forest Anomaly", marker=dict(color="crimson", size=12, symbol="triangle-up")
    ))
elif method == "Z-Score (rolling)":
    fig.add_trace(go.Scatter(
        x=weekly["Week"], y=weekly["rolling_mean"], mode="lines",
        name="8-Week Rolling Mean", line=dict(color="gray", width=1, dash="dash")
    ))
    anomalies = weekly[weekly["z_anomaly"]]
    fig.add_trace(go.Scatter(
        x=anomalies["Week"], y=anomalies["Weekly Sales"], mode="markers",
        name="Z-Score Anomaly", marker=dict(color="darkorange", size=12, symbol="diamond")
    ))
else:
    both = weekly[weekly["iso_anomaly"] & weekly["z_anomaly"]]
    iso_only = weekly[weekly["iso_anomaly"] & ~weekly["z_anomaly"]]
    z_only = weekly[~weekly["iso_anomaly"] & weekly["z_anomaly"]]
    fig.add_trace(go.Scatter(x=both["Week"], y=both["Weekly Sales"], mode="markers",
                              name=f"Both methods ({len(both)})",
                              marker=dict(color="black", size=16, symbol="star")))
    fig.add_trace(go.Scatter(x=iso_only["Week"], y=iso_only["Weekly Sales"], mode="markers",
                              name=f"Isolation Forest only ({len(iso_only)})",
                              marker=dict(color="crimson", size=12, symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=z_only["Week"], y=z_only["Weekly Sales"], mode="markers",
                              name=f"Z-Score only ({len(z_only)})",
                              marker=dict(color="darkorange", size=12, symbol="diamond")))

fig.update_layout(
    title="Weekly Sales — Anomaly Detection",
    xaxis_title="Week", yaxis_title="Weekly Sales ($)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Table of detected anomalies
# ---------------------------------------------------------------------------
st.subheader("Detected Anomaly Dates")

anomaly_table = weekly[weekly["iso_anomaly"] | weekly["z_anomaly"]].copy()
anomaly_table["Flagged By"] = anomaly_table.apply(
    lambda r: ", ".join(filter(None, [
        "Isolation Forest" if r["iso_anomaly"] else None,
        "Z-Score" if r["z_anomaly"] else None
    ])), axis=1
)
anomaly_table["Week"] = anomaly_table["Week"].dt.strftime("%Y-%m-%d")
anomaly_table["Weekly Sales"] = anomaly_table["Weekly Sales"].map(lambda v: f"${v:,.2f}")

display_cols = anomaly_table[["Week", "Weekly Sales", "Flagged By"]].sort_values("Week")
st.dataframe(display_cols, use_container_width=True, hide_index=True)

st.caption(
    f"Isolation Forest flagged {weekly['iso_anomaly'].sum()} weeks (global outliers relative to the "
    f"full 4-year distribution). The Z-score method flagged {weekly['z_anomaly'].sum()} weeks "
    "(local jumps relative to each week's own 8-week neighborhood). Only 2 weeks were flagged by "
    "both — see Task 5 in the analysis notebook for the full explanation of why these methods "
    "largely disagree, and what each one is actually useful for."
)
