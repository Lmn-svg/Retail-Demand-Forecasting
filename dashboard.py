import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Retail Demand Forecasting Dashboard",
    layout="wide"
)

# ============================================
# Dashboard Title
# ============================================

st.title("Retail Demand Forecasting Dashboard")

st.markdown(
    """
    This dashboard provides retail sales monitoring,
    forecasting insights, and business recommendations
    using the Random Forest forecasting model.
    """
)

# ============================================
# Load Dataset
# ============================================

@st.cache_data
def load_data():
    df = pd.read_csv("dashboard_data.csv")
    
    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

df = load_data()

# ============================================
# Sidebar Filters
# ============================================

st.sidebar.header("Filters")

# Store filter
selected_store = st.sidebar.multiselect(
    "Select Store",
    sorted(df['Store'].unique()),
    default=sorted(df['Store'].unique())[:5]
)

# Date filter
min_date = df['Date'].min()
max_date = df['Date'].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date]
)

# Sales threshold slider
sales_threshold = st.sidebar.slider(
    "Minimum Weekly Sales",
    min_value=0,
    max_value=int(df['Weekly_Sales'].max()),
    value=10000
)

# ============================================
# Filter Dataset
# ============================================

filtered_df = df[
    (df['Store'].isin(selected_store)) &
    (df['Date'] >= pd.to_datetime(date_range[0])) &
    (df['Date'] <= pd.to_datetime(date_range[1])) &
    (df['Weekly_Sales'] >= sales_threshold)
]

# ============================================
# KPI Calculations
# ============================================

total_sales = filtered_df['Weekly_Sales'].sum()

forecast_sales = filtered_df['rolling_mean_4'].mean()

sales_growth = (
    filtered_df['Weekly_Sales'].pct_change().mean() * 100
)

forecast_accuracy = round(
    100 - 15.7,
    1
)

if len(filtered_df) > 0:

    top_store = (
        filtered_df.groupby('Store')['Weekly_Sales']
        .sum()
        .idxmax()
    )

else:

    top_store = "N/A"

# ============================================
# KPI Section
# ============================================

st.subheader("Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Sales",
    f"${total_sales:,.0f}"
)

col2.metric(
    "Forecasted Sales",
    f"${forecast_sales:,.0f}"
)

col3.metric(
    "Sales Growth",
    f"{sales_growth:.2f}%"
)

col4.metric(
    "Forecast Accuracy",
    f"{forecast_accuracy:.1f}%"
)

col5.metric(
    "Top Store",
    f"Store {top_store}"
)

# ============================================
# Main Layout
# ============================================

left_col, right_col = st.columns([1, 2])

# ============================================
# Left Side Charts
# ============================================
with left_col:

    st.subheader("Top 10 Stores")

    top_stores = (
        filtered_df
        .groupby('Store')['Weekly_Sales']
        .sum()
        .reset_index()
        .sort_values(
            by='Weekly_Sales',
            ascending=False
        )
        .head(10)
    )

    fig_top = px.bar(
        top_stores,
        x='Store',
        y='Weekly_Sales',
        title='Top 10 Stores'
    )

    st.plotly_chart(
        fig_top,
        use_container_width=True
    )

    st.subheader("Bottom 10 Stores")

    bottom_stores = (
        filtered_df
        .groupby('Store')['Weekly_Sales']
        .sum()
        .reset_index()
        .sort_values(
            by='Weekly_Sales',
            ascending=True
        )
        .head(10)
    )

    fig_bottom = px.bar(
        bottom_stores,
        x='Store',
        y='Weekly_Sales',
        title='Bottom 10 Stores'
    )

    st.plotly_chart(
        fig_bottom,
        use_container_width=True
    )

    st.subheader("Store Performance")

    store_sales = (
        filtered_df.groupby('Store')['Weekly_Sales']
        .sum()
        .reset_index()
    )

    fig_bar = px.bar(
        store_sales,
        x='Store',
        y='Weekly_Sales',
        title='Sales by Store'
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

    st.subheader("Monthly Sales Trend")

    monthly_sales = (
        filtered_df.groupby('Month')['Weekly_Sales']
        .mean()
        .reset_index()
    )

    fig_month = px.line(
        monthly_sales,
        x='Month',
        y='Weekly_Sales',
        markers=True,
        title='Average Monthly Sales'
    )

    st.plotly_chart(
        fig_month,
        use_container_width=True
    )

# ============================================
# Right Side Main Visualization
# ============================================

with right_col:

    st.subheader("Actual vs Forecasted Sales")

    forecast_df = filtered_df.copy()

    forecast_df['Predicted_Sales'] = (
        forecast_df['rolling_mean_4']
    )

    fig_main = go.Figure()

    fig_main.add_trace(
        go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Weekly_Sales'],
            mode='lines',
            name='Actual Sales'
        )
    )

    fig_main.add_trace(
        go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Predicted_Sales'],
            mode='lines',
            name='Predicted Sales'
        )
    )

    fig_main.update_layout(
        title='Sales Forecasting',
        xaxis_title='Date',
        yaxis_title='Weekly Sales'
    )

    st.plotly_chart(
        fig_main,
        use_container_width=True
    )
# ============================================
# Feature Importance
# ============================================

st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    'Feature': [
        'rolling_mean_4',
        'rolling_std_4',
        'CPI',
        'Unemployment',
        'Fuel_Price',
        'Temperature'
    ],
    'Importance': [
        0.884,
        0.101,
        0.003,
        0.002,
        0.001,
        0.001
    ]
})

fig_importance = px.bar(
    importance_df,
    x='Feature',
    y='Importance',
    title='Random Forest Feature Importance'
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)

# ============================================
# Holiday Impact Analysis
# ============================================

st.subheader("Holiday Impact Analysis")

holiday_df = (
    filtered_df
    .groupby('IsHoliday')['Weekly_Sales']
    .mean()
    .reset_index()
)

holiday_df['IsHoliday'] = (
    holiday_df['IsHoliday']
    .replace({
        False: 'Non-Holiday',
        True: 'Holiday'
    })
)

fig_holiday = px.bar(
    holiday_df,
    x='IsHoliday',
    y='Weekly_Sales',
    title='Average Sales: Holiday vs Non-Holiday'
)

st.plotly_chart(
    fig_holiday,
    use_container_width=True
)

fig_holiday = px.bar(
    holiday_sales,
    x='IsHoliday',
    y='Weekly_Sales',
    title='Average Sales: Holiday vs Non-Holiday'
)

st.plotly_chart(
    fig_holiday,
    use_container_width=True
)
# ============================================
# Error Distribution
# ============================================

st.subheader("Prediction Error Distribution")

filtered_df = filtered_df.copy()

filtered_df['Prediction_Error'] = (
    filtered_df['Weekly_Sales']
    - filtered_df['rolling_mean_4']
)
fig_error = px.histogram(
    filtered_df,
    x='Prediction_Error',
    nbins=50,
    title='Residual Distribution'
)

st.plotly_chart(
    fig_error,
    use_container_width=True
)

# ============================================
# Recommendations Section
# ============================================
st.subheader("Store Performance Alert")

current_sales = (
    filtered_df['Weekly_Sales']
    .mean()
)

overall_sales = (
    df['Weekly_Sales']
    .mean()
)

ratio = current_sales / overall_sales

if ratio >= 0.95:

    st.success(
        "🟢 Normal Performance: Sales are within expected range."
    )

elif ratio >= 0.80:

    st.warning(
        "🟡 Warning: Sales are below average and should be monitored."
    )

else:

    st.error(
        "🔴 Critical Alert: Sales are significantly below expected levels."
    )
st.subheader("Business Recommendations")

# Recommendation logic
avg_sales = filtered_df['Weekly_Sales'].mean()

holiday_sales = (
    filtered_df[filtered_df['IsHoliday'] == True]
    ['Weekly_Sales']
    .mean()
)

if holiday_sales > avg_sales:
    st.success(
        "Increase inventory levels before holiday periods due to higher expected demand."
    )

low_store = (
    filtered_df.groupby('Store')['Weekly_Sales']
    .mean()
    .idxmin()
)

st.warning(
    f"Store {low_store} shows relatively weak performance and may require promotional support."
)

high_variance = filtered_df['rolling_std_4'].mean()

if high_variance > 5000:
    st.info(
        "Sales volatility is currently high. Monitor demand fluctuations carefully."
    )

# ============================================
# Footer
# ============================================

st.markdown("---")

st.markdown(
    """
    Retail Demand Forecasting Dashboard  
    Powered by Random Forest Forecasting Model
    """
)
