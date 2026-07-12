# Sales Intelligence Dashboard

A 4-page Streamlit dashboard built on top of the Superstore Sales forecasting/anomaly/segmentation
analysis (Tasks 1–6). It's split into:

- **`app.py`** — Page 1: Sales Overview Dashboard (year/region/category filters, KPIs, trend charts)
- **`pages/1_📈_Forecast_Explorer.py`** — Page 2: Prophet forecasts by category/region with MAE/RMSE
- **`pages/2_🚨_Anomaly_Report.py`** — Page 3: Isolation Forest + Z-score anomaly detection
- **`pages/3_🧩_Product_Demand_Segments.py`** — Page 4: K-Means product demand clusters

## How it's structured (and why)

All the heavy modeling (Prophet forecasts, Isolation Forest, K-Means/PCA) runs **once**, offline, via
`build_artifacts.py`, which writes lightweight CSVs into `data/`:

- `data/forecasts.csv` — historical monthly actuals + 3-month-ahead forecasts, per segment
- `data/forecast_metrics.csv` — backtested MAE/RMSE/MAPE per segment
- `data/anomalies.csv` — weekly sales with Isolation Forest & Z-score anomaly flags
- `data/clusters.csv` — sub-category features, cluster assignments, labels, PCA coordinates

The Streamlit app itself only reads these CSVs and does lightweight pandas filtering — it never
retrains a model live. This is a deliberate production choice: Prophet's compiled Stan backend is slow
to fit and can be finicky to install on free hosting tiers, and no one wants a dashboard that hangs for
30 seconds every time a filter changes.

**If the underlying sales data changes**, re-run `python build_artifacts.py` locally (needs
`requirements-dev.txt` installed) and commit the refreshed CSVs — the deployed app will pick them up
on its next restart. The app's own `requirements.txt` stays lightweight (streamlit, pandas, plotly only).

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Use the sidebar page list (or the emoji-named files under `pages/`)
to navigate between the 4 pages.

## Deploy on Streamlit Community Cloud (free)

I can't create a GitHub repository or a Streamlit Cloud account on your behalf, so here's exactly what
to do — it takes about 5 minutes:

1. **Create a GitHub repo** (e.g. `sales-dashboard`) and push this entire folder to it, including the
   `data/` folder with all four CSVs already generated (don't skip this — the app reads them at
   startup and has nothing to show without them).
   ```bash
   cd streamlit_app
   git init
   git add .
   git commit -m "Sales intelligence dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/sales-dashboard.git
   git push -u origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account.
3. Click **"New app"**, select the repo you just pushed, branch `main`, and set the main file path to
   `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically and the app goes live at
   a URL like `https://<your-app-name>.streamlit.app`.
5. That URL is your live link to share — the app updates automatically every time you push to `main`.

## File checklist before pushing

```
streamlit_app/
├── app.py
├── build_artifacts.py
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── pages/
│   ├── 1_📈_Forecast_Explorer.py
│   ├── 2_🚨_Anomaly_Report.py
│   └── 3_🧩_Product_Demand_Segments.py
└── data/
    ├── superstore_sales.csv
    ├── forecasts.csv
    ├── forecast_metrics.csv
    ├── anomalies.csv
    └── clusters.csv
```
