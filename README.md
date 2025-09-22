# Streamlit Forecast Dashboard

A beginner-friendly Streamlit app to visualize and evaluate **Actual vs Predicted** values from CSV/Excel files.

## Features
- Upload CSV/Excel (multiple sheets supported)
- Pick which columns are *Actual* and *Predicted*
- Metrics: R2, RMSE, MAE, MAPE
- Charts: overlay line, scatter with trendline, residual histogram
- Download metrics and tidy data

## Local Run (no Git needed)
1. Install Python 3.10+.
2. Open a terminal in this folder and run:
   ```bash
   python -m venv .venv
   . .venv/bin/activate  # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. In the browser, upload your file.

## Deploy on Streamlit Community Cloud
1. Push this folder to a **public GitHub repo**.
2. Go to **share.streamlit.io** → **New app**.
3. Select your repo, branch (`main`), and `app.py` as the entry point.
4. Press **Deploy**.

## Notes
- A copy of your uploaded Excel file is in data/Actual_vs_Predicted_Results.xlsx for local testing. Remove it before pushing to GitHub if sensitive.
- If your data is sensitive, consider keeping the repo **private** on GitHub.
