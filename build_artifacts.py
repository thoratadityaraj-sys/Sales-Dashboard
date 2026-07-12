"""
Precomputes all heavy analysis (Prophet forecasts, anomaly detection, clustering)
ONE TIME and saves results to lightweight CSVs in data/.

Why precompute instead of running models live in the Streamlit app?
- Prophet has a compiled Stan backend that is slow to fit and can be finicky to
  install on Streamlit Community Cloud's free tier.
- Re-fitting 6+ models on every page load / filter change would make the app feel
  broken (10-30s freezes). A production dashboard should read pre-scored results
  and only do lightweight pandas filtering live.

Run this once locally: `python build_artifacts.py`
Re-run it whenever the underlying data changes.
"""
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DATA_PATH = 'data/superstore_sales.csv'

print("Loading raw data...")
df = pd.read_csv(DATA_PATH)
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['YearMonth'] = df['Order Date'].dt.to_period('M')

full_date_range = pd.date_range(df['Order Date'].min(), df['Order Date'].max(), freq='D')

# ---------------------------------------------------------------------------
# 1. FORECASTS (Prophet, per segment) - both a backtest (for MAE/RMSE) and a
#    true future 3-month forecast (for the chart)
# ---------------------------------------------------------------------------
print("\nBuilding per-segment Prophet forecasts (this takes a minute or two)...")

def build_daily_series(sub_df):
    s = sub_df.groupby('Order Date')['Sales'].sum()
    s = s.reindex(full_date_range, fill_value=0).reset_index()
    s.columns = ['ds', 'y']
    return s

def fit_prophet_forecast(daily_df, periods, future_start=None):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                     seasonality_mode='multiplicative')
    model.fit(daily_df)
    future = model.make_future_dataframe(periods=periods, freq='D')
    fcst = model.predict(future)
    if future_start is not None:
        fcst = fcst[fcst['ds'] >= future_start]
    fcst = fcst[['ds', 'yhat']].copy()
    fcst['Month'] = fcst['ds'].values.astype('datetime64[M]')
    return fcst.groupby('Month')['yhat'].sum()

segments = {
    'Overall':                    df,
    'Furniture (Category)':       df[df['Category'] == 'Furniture'],
    'Technology (Category)':      df[df['Category'] == 'Technology'],
    'Office Supplies (Category)': df[df['Category'] == 'Office Supplies'],
    'West (Region)':              df[df['Region'] == 'West'],
    'East (Region)':              df[df['Region'] == 'East'],
}

test_cutoff = pd.Timestamp('2018-10-01')
test_end = pd.Timestamp('2018-12-31')
backtest_horizon = (test_end - df['Order Date'].max()).days  # negative-safe: recompute below properly

future_start = pd.Timestamp('2019-01-01')
future_end = pd.Timestamp('2019-03-31')
future_horizon = (future_end - df['Order Date'].max()).days

forecast_rows = []
metrics_rows = []

for name, sub in segments.items():
    print(f"  Fitting: {name}")
    daily = build_daily_series(sub)

    # --- Backtest: train on data before Oct 2018, forecast Oct-Dec 2018, compare to actual
    train_daily = daily[daily['ds'] < test_cutoff]
    horizon_days_test = (test_end - train_daily['ds'].max()).days
    monthly_pred_test = fit_prophet_forecast(train_daily, horizon_days_test, future_start=test_cutoff)

    actual_test = daily[(daily['ds'] >= test_cutoff) & (daily['ds'] <= test_end)].copy()
    actual_test['Month'] = actual_test['ds'].values.astype('datetime64[M]')
    monthly_actual_test = actual_test.groupby('Month')['y'].sum()

    common_months = monthly_pred_test.index.intersection(monthly_actual_test.index)
    mae = np.mean(np.abs(monthly_actual_test[common_months].values - monthly_pred_test[common_months].values))
    rmse = np.sqrt(np.mean((monthly_actual_test[common_months].values - monthly_pred_test[common_months].values) ** 2))
    mape = np.mean(np.abs((monthly_actual_test[common_months].values - monthly_pred_test[common_months].values)
                           / monthly_actual_test[common_months].values)) * 100
    metrics_rows.append({'Segment': name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape})

    # --- True future forecast: train on ALL history, forecast Jan-Mar 2019
    monthly_pred_future = fit_prophet_forecast(daily, future_horizon, future_start=future_start)

    # Save historical monthly actuals (for chart context) + future forecast
    hist_monthly = daily.set_index('ds')['y'].resample('MS').sum()
    for month, val in hist_monthly.items():
        forecast_rows.append({'Segment': name, 'Month': month, 'Value': val, 'Type': 'Actual'})
    for month, val in monthly_pred_future.items():
        forecast_rows.append({'Segment': name, 'Month': month, 'Value': val, 'Type': 'Forecast'})

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv('data/forecasts.csv', index=False)
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv('data/forecast_metrics.csv', index=False)
print(f"Saved data/forecasts.csv ({len(forecast_df)} rows) and data/forecast_metrics.csv")

# ---------------------------------------------------------------------------
# 2. ANOMALY DETECTION (weekly, Isolation Forest + Z-score)
# ---------------------------------------------------------------------------
print("\nRunning anomaly detection...")
weekly = df.set_index('Order Date').resample('W')['Sales'].sum().reset_index()
weekly.columns = ['Week', 'Weekly Sales']

iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
weekly['iso_anomaly'] = (iso.fit_predict(weekly[['Weekly Sales']]) == -1)

window = 8
weekly['rolling_mean'] = weekly['Weekly Sales'].rolling(window, min_periods=4, center=True).mean()
weekly['rolling_std'] = weekly['Weekly Sales'].rolling(window, min_periods=4, center=True).std()
weekly['z_score'] = (weekly['Weekly Sales'] - weekly['rolling_mean']) / weekly['rolling_std']
weekly['z_anomaly'] = weekly['z_score'].abs() > 2

weekly.to_csv('data/anomalies.csv', index=False)
print(f"Saved data/anomalies.csv ({len(weekly)} weeks, "
      f"{weekly['iso_anomaly'].sum()} IsoForest anomalies, {weekly['z_anomaly'].sum()} Z-score anomalies)")

# ---------------------------------------------------------------------------
# 3. PRODUCT DEMAND CLUSTERING (sub-category level)
# ---------------------------------------------------------------------------
print("\nRunning product demand clustering...")
total_sales = df.groupby('Sub-Category')['Sales'].sum()
yearly_by_subcat = df.groupby(['Sub-Category', 'Year'])['Sales'].sum().unstack('Year')
cagr_pct = ((yearly_by_subcat[2018] / yearly_by_subcat[2015]) ** (1 / 3) - 1) * 100
monthly_by_subcat = df.groupby(['Sub-Category', 'YearMonth'])['Sales'].sum().unstack('YearMonth')
volatility = monthly_by_subcat.std(axis=1)
avg_order_value = df.groupby('Sub-Category')['Sales'].mean()

subcat_features = pd.DataFrame({
    'Total Sales Volume': total_sales,
    'YoY Growth Rate (%)': cagr_pct,
    'Sales Volatility': volatility,
    'Avg Order Value': avg_order_value
})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(subcat_features)

kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
subcat_features['Cluster'] = kmeans_final.fit_predict(X_scaled)

cluster_summary = subcat_features.groupby('Cluster')[
    ['Total Sales Volume', 'YoY Growth Rate (%)', 'Sales Volatility']].mean()
overall_median = subcat_features[['Total Sales Volume', 'YoY Growth Rate (%)', 'Sales Volatility']].median()

def label_cluster(row):
    growth_tag = None
    if row['YoY Growth Rate (%)'] < 0:
        growth_tag = 'Declining Demand'
    elif row['YoY Growth Rate (%)'] > overall_median['YoY Growth Rate (%)'] * 2:
        growth_tag = 'Growing Demand'
    volume_tag = 'High Volume' if row['Total Sales Volume'] >= overall_median['Total Sales Volume'] else 'Low Volume'
    volatility_tag = 'High Volatility' if row['Sales Volatility'] >= overall_median['Sales Volatility'] else 'Stable Demand'
    if growth_tag:
        return f"{growth_tag} ({volume_tag})"
    return f"{volume_tag}, {volatility_tag}"

cluster_summary['Label'] = cluster_summary.apply(label_cluster, axis=1)
label_map = cluster_summary['Label'].to_dict()
subcat_features['Cluster Label'] = subcat_features['Cluster'].map(label_map)

pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_scaled)
subcat_features['PC1'] = pca_coords[:, 0]
subcat_features['PC2'] = pca_coords[:, 1]

subcat_features = subcat_features.reset_index().rename(columns={'index': 'Sub-Category'})
subcat_features.to_csv('data/clusters.csv', index=False)
print(f"Saved data/clusters.csv ({len(subcat_features)} sub-categories, 4 clusters)")

print("\nAll artifacts built successfully.")
