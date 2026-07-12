import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Product Demand Segments", page_icon="🧩", layout="wide")

st.title("🧩 Product Demand Segments")
st.caption("Page 4 of 4 — Sub-categories clustered by demand behavior using K-Means (Task 6).")


@st.cache_data
def load_clusters():
    return pd.read_csv("data/clusters.csv")


clusters = load_clusters()

cluster_colors = {
    "Growing Demand (High Volume)": "#d62728",
    "Declining Demand (High Volume)": "#ff7f0e",
    "High Volume, High Volatility": "#2ca02c",
    "Low Volume, Stable Demand": "#1f77b4",
}

# ---------------------------------------------------------------------------
# PCA scatter plot
# ---------------------------------------------------------------------------
st.subheader("Cluster Map (PCA-Reduced to 2D)")

fig = px.scatter(
    clusters, x="PC1", y="PC2", color="Cluster Label", text="Sub-Category",
    color_discrete_map=cluster_colors, size="Total Sales Volume", size_max=40,
    hover_data={"Total Sales Volume": ":$,.0f", "YoY Growth Rate (%)": ":.1f",
                "Sales Volatility": ":$,.0f", "Avg Order Value": ":$,.2f",
                "PC1": False, "PC2": False}
)
fig.update_traces(textposition="top center")
fig.update_layout(
    xaxis_title="PC1", yaxis_title="PC2",
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Sub-categories by cluster table
# ---------------------------------------------------------------------------
st.subheader("Sub-Categories by Demand Cluster")

selected_label = st.selectbox("Filter by cluster:", ["All"] + list(clusters["Cluster Label"].unique()))

table_data = clusters if selected_label == "All" else clusters[clusters["Cluster Label"] == selected_label]
table_data = table_data.sort_values(["Cluster Label", "Total Sales Volume"], ascending=[True, False]).copy()

table_data["Total Sales Volume"] = table_data["Total Sales Volume"].map(lambda v: f"${v:,.0f}")
table_data["YoY Growth Rate (%)"] = table_data["YoY Growth Rate (%)"].map(lambda v: f"{v:.1f}%")
table_data["Sales Volatility"] = table_data["Sales Volatility"].map(lambda v: f"${v:,.0f}")
table_data["Avg Order Value"] = table_data["Avg Order Value"].map(lambda v: f"${v:,.2f}")

st.dataframe(
    table_data[["Sub-Category", "Cluster Label", "Total Sales Volume", "YoY Growth Rate (%)",
                "Sales Volatility", "Avg Order Value"]],
    use_container_width=True, hide_index=True
)

st.divider()

# ---------------------------------------------------------------------------
# Recommended stocking strategy
# ---------------------------------------------------------------------------
st.subheader("Recommended Stocking Strategy")

strategy_text = {
    "Growing Demand (High Volume)": (
        "**Increase safety stock proactively.** Demand is accelerating fast (~80% CAGR for Copiers) — "
        "last year's inventory policy will already be too thin. High order value + high volatility "
        "means a missed reorder point risks a costly stockout, so favor buffer stock over lean/JIT ordering."
    ),
    "Low Volume, Stable Demand": (
        "**Lean, low-buffer inventory with infrequent batch reorders.** Demand is small and predictable, "
        "so there's little upside to holding much safety stock. A simple monthly reorder review is sufficient."
    ),
    "High Volume, High Volatility": (
        "**Priority segment for active forecasting.** These sub-categories drive the bulk of revenue but "
        "swing meaningfully month-to-month. Use per-category forecasts (see Forecast Explorer) to set "
        "dynamic reorder points rather than a fixed static buffer."
    ),
    "Declining Demand (High Volume)": (
        "**Wind down inventory deliberately.** Machines is the only sub-category with negative YoY growth "
        "despite high volume/value. Sell through existing stock rather than replenishing at historical "
        "levels; consider promotions to clear inventory."
    ),
}

for label, text in strategy_text.items():
    with st.expander(f"📦 {label}"):
        st.markdown(text)
