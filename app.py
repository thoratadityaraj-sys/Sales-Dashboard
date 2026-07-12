import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sales Intelligence Dashboard", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Data loading (cached so it only happens once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/superstore_sales.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    return df

df = load_data()

st.title("📊 Sales Overview Dashboard")
st.caption("Superstore Sales — Intelligent Forecasting System | Page 1 of 4")

# ---------------------------------------------------------------------------
# Sidebar filters (shared across all charts on this page)
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

years = sorted(df["Year"].unique())
selected_years = st.sidebar.multiselect("Year", years, default=years)

regions = sorted(df["Region"].unique())
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)

categories = sorted(df["Category"].unique())
selected_categories = st.sidebar.multiselect("Category", categories, default=categories)

filtered = df[
    df["Year"].isin(selected_years)
    & df["Region"].isin(selected_regions)
    & df["Category"].isin(selected_categories)
]

if filtered.empty:
    st.warning("No data matches the current filter selection. Adjust the filters in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"${filtered['Sales'].sum():,.0f}")
col2.metric("Total Orders", f"{filtered['Order ID'].nunique():,}")
col3.metric("Avg Order Value", f"${filtered.groupby('Order ID')['Sales'].sum().mean():,.2f}")
col4.metric("Sub-Categories", f"{filtered['Sub-Category'].nunique()}")

st.divider()

# ---------------------------------------------------------------------------
# Total sales by year (bar chart)
# ---------------------------------------------------------------------------
st.subheader("Total Sales by Year")
yearly_sales = filtered.groupby("Year", as_index=False)["Sales"].sum()
fig_year = px.bar(yearly_sales, x="Year", y="Sales", text_auto=".2s", color="Sales",
                   color_continuous_scale="Blues")
fig_year.update_layout(xaxis=dict(dtick=1), yaxis_title="Total Sales ($)", coloraxis_showscale=False)
st.plotly_chart(fig_year, use_container_width=True)

# ---------------------------------------------------------------------------
# Monthly sales trend (line chart)
# ---------------------------------------------------------------------------
st.subheader("Monthly Sales Trend")
monthly_sales = filtered.groupby("Month", as_index=False)["Sales"].sum()
fig_month = px.line(monthly_sales, x="Month", y="Sales", markers=True)
fig_month.update_layout(yaxis_title="Sales ($)", xaxis_title="Month")
st.plotly_chart(fig_month, use_container_width=True)

# ---------------------------------------------------------------------------
# Sales by region and category (interactive)
# ---------------------------------------------------------------------------
st.subheader("Sales by Region and Category")

view = st.radio("View as:", ["Grouped Bar Chart", "Heatmap"], horizontal=True)

region_cat = filtered.groupby(["Region", "Category"], as_index=False)["Sales"].sum()

if view == "Grouped Bar Chart":
    fig_rc = px.bar(region_cat, x="Region", y="Sales", color="Category", barmode="group",
                     text_auto=".2s")
    fig_rc.update_layout(yaxis_title="Sales ($)")
    st.plotly_chart(fig_rc, use_container_width=True)
else:
    pivot = region_cat.pivot(index="Category", columns="Region", values="Sales").fillna(0)
    fig_rc = px.imshow(pivot, text_auto=".2s", color_continuous_scale="Blues", aspect="auto")
    fig_rc.update_layout(xaxis_title="Region", yaxis_title="Category")
    st.plotly_chart(fig_rc, use_container_width=True)

with st.expander("View underlying data"):
    st.dataframe(region_cat.sort_values("Sales", ascending=False), use_container_width=True)

st.sidebar.divider()
st.sidebar.info("Use the page navigation above to explore Forecasts, Anomalies, and Product Segments.")
