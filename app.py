import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------- Page setup ----------
st.set_page_config(page_title='Forecast Dashboard', layout='wide')
st.title('📈 Forecast Dashboard (Actual vs Predicted)')

st.markdown('''
This app lets you upload a CSV or Excel file and interactively visualize/assess **Actual vs Predicted** results.

- Works with multiple Excel sheets  
- Lets you pick which columns are *Actual* and *Predicted*  
- Computes metrics (R2, RMSE, MAE, MAPE)  
- Plots overlay line chart, scatter plot (with fitted line), and residual diagnostics
''')

with st.expander('ℹ️ How to format your data'):
    st.write('''
    Your file should contain at least two **numeric** columns representing **Actual** and **Predicted** values.
    A date or index column is optional but helpful for plotting.
    ''')

# ---------- Data helpers ----------
def load_data_from_bytes(file):
    name = file.name.lower()
    if name.endswith('.csv'):
        return {'Sheet1': pd.read_csv(file)}
    else:
        # Excel: load all sheets
        return pd.read_excel(file, sheet_name=None, engine='openpyxl')

def load_sample():
    """Optional sample file from a local 'data/' folder if you kept one."""
    try:
        return pd.read_excel('data/Actual_vs_Predicted_Results.xlsx', sheet_name=None, engine='openpyxl')
    except Exception as e:
        st.error(f"Couldn't load sample file from data/: {e}")
        return None

# ---------- File input ----------
uploaded = st.file_uploader('Upload CSV or Excel file', type=['csv', 'xlsx', 'xls'])

data_dict = None
if uploaded is not None:
    try:
        data_dict = load_data_from_bytes(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
else:
    st.info('No file uploaded yet. You can also use a sample file from **data/** if provided.')
    if st.button('Load sample from data/'):
        data_dict = load_sample()

if data_dict is None:
    st.stop()

# ---------- Sheet selection (for Excel) ----------
sheet_name = st.selectbox('Select sheet', list(data_dict.keys()))
df = data_dict[sheet_name].copy()

st.subheader('Preview')
st.dataframe(df.head(50), use_container_width=True)

# ---------- Choose columns ----------
# Find numeric columns for Actual/Predicted
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
if len(numeric_cols) < 2:
    st.warning("I couldn't find at least two numeric columns. Please upload a file with numeric Actual & Predicted columns.")
    st.stop()

x_cols = df.columns.tolist()
x_col = st.selectbox('Optional: choose an X-axis column (e.g., Date/Index). Leave empty for row index.',
                     ['<row index>'] + x_cols)

left, right = st.columns(2)
with left:
    actual_col = st.selectbox('Actual column', numeric_cols, index=0 if len(numeric_cols) > 0 else None)
with right:
    pred_col = st.selectbox('Predicted column', numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

if actual_col == pred_col:
    st.error("Actual and Predicted columns must be different.")
    st.stop()

# Subset & clean
df_metrics = df[[actual_col, pred_col]].dropna().copy()
if df_metrics.empty:
    st.warning("After dropping missing values, there are no rows left to analyze.")
    st.stop()

if x_col != '<row index>':
    # carry over the chosen x-axis column (no coercion needed; Plotly can render strings/datetimes)
    df_metrics[x_col] = df.loc[df_metrics.index, x_col]

# ---------- Metrics ----------
y_true = df_metrics[actual_col].to_numpy()
y_pred = df_metrics[pred_col].to_numpy()

r2 = float(r2_score(y_true, y_pred))
rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
mae = float(mean_absolute_error(y_true, y_pred))

# Robust MAPE: ignore zeros in y_true
valid = y_true != 0
if np.any(valid):
    mape = float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / np.abs(y_true[valid]))) * 100)
else:
    mape = np.nan

m1, m2, m3, m4 = st.columns(4)
m1.metric('R2', f'{r2:.4f}')
m2.metric('RMSE', f'{rmse:,.4f}')
m3.metric('MAE', f'{mae:,.4f}')
m4.metric('MAPE (%)', '—' if np.isnan(mape) else f'{mape:,.2f}')

st.markdown('---')

# ---------- Overlay line chart (Actual vs Predicted over X) ----------
if x_col != '<row index>':
    x_vals = df_metrics[x_col]
else:
    x_vals = df_metrics.index

line_df = pd.DataFrame({
    "x": x_vals,
    "Actual": df_metrics[actual_col].values,
    "Predicted": df_metrics[pred_col].values
})
fig_line = px.line(line_df, x='x', y=['Actual', 'Predicted'], title='Actual vs Predicted (Overlay)')
fig_line.update_layout(legend_title_text='')
st.plotly_chart(fig_line, use_container_width=True)

# ---------- Scatter with fitted line (no statsmodels needed) ----------
scatter_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})

# Compute simple linear fit y = a*x + b using numpy
try:
    a, b = np.polyfit(scatter_df['Actual'], scatter_df['Predicted'], 1)
    x_min, x_max = scatter_df['Actual'].min(), scatter_df['Actual'].max()
    x_fit = np.linspace(x_min, x_max, 100)
    y_fit = a * x_fit + b
except Exception:
    a = b = None
    x_fit = y_fit = None

fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=scatter_df['Actual'], y=scatter_df['Predicted'],
    mode='markers', name='Points'
))
# Perfect-fit reference line (y = x)
x_ref = np.linspace(scatter_df['Actual'].min(), scatter_df['Actual'].max(), 2)
fig_scatter.add_trace(go.Scatter(
    x=x_ref, y=x_ref, mode='lines', name='Perfect Fit (y = x)', line=dict(dash='dash')
))
# Fitted regression line
if x_fit is not None:
    fig_scatter.add_trace(go.Scatter(
        x=x_fit, y=y_fit, mode='lines', name='Fitted Line'
))
fig_scatter.update_layout(title='Scatter: Actual vs Predicted', xaxis_title='Actual', yaxis_title='Predicted')
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------- Residuals ----------
resid = y_true - y_pred
resid_df = pd.DataFrame({'Residual': resid})
fig_resid = px.histogram(resid_df, x='Residual', nbins=40, title='Residual Distribution')
st.plotly_chart(fig_resid, use_container_width=True)

# ---------- Downloads ----------
st.subheader('Downloads')

metrics_df = pd.DataFrame(
    {'metric': ['R2', 'RMSE', 'MAE', 'MAPE'],
     'value': [r2, rmse, mae, mape]}
)
st.dataframe(metrics_df, use_container_width=True)

csv_bytes = metrics_df.to_csv(index=False).encode('utf-8')
st.download_button('Download metrics CSV', data=csv_bytes, file_name='metrics.csv')

tidy_df = df_metrics.copy()
tidy_df.rename(columns={actual_col: 'Actual', pred_col: 'Predicted'}, inplace=True)
if x_col != '<row index>':
    tidy_cols = [x_col, 'Actual', 'Predicted']
else:
    tidy_cols = ['Actual', 'Predicted']
tidy_bytes = tidy_df[tidy_cols].to_csv(index=False).encode('utf-8')
st.download_button('Download tidy data CSV', data=tidy_bytes, file_name='tidy_actual_vs_predicted.csv')

st.caption('Built with Streamlit • Plotly • pandas • scikit-learn')
