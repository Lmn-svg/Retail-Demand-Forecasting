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
# Language Selector
# ============================================

col_title, col_lang = st.columns([8,1])

with col_lang:
    language = st.selectbox(
        "",
        ["English", "中文"]
    )


# ============================================
# Language Selector
# ============================================

translations = {

    "English": {

        "store": "Store",
        
        "Weekly_sales": "Weekly sales",
        

        "Introduction":
        "This dashboard provides retail sales monitoring, forecasting insights, and business recommendations using the Random Forest forecasting model.",
        
        "title":
        "Retail Demand Forecasting Dashboard",

        "description":
        "This dashboard provides retail sales monitoring, forecasting insights, and business recommendations using the Random Forest forecasting model.",

        "filters": "Filters",
        "select_store": "Select Store",
        "select_date": "Select Date Range",
        "minimum_sales": "Minimum Weekly Sales",

        "kpi": "Key Performance Indicators",

        "total_sales": "Total Sales",
        "forecast_sales": "Forecasted Sales",
        "sales_growth": "Sales Growth",
        "forecast_accuracy": "Forecast Accuracy",
        "top_store": "Top Store",

        "top10": "Top 10 Stores",
        "bottom10": "Bottom 10 Stores",
        "store_performance": "Store Performance",
        "monthly_sales": "Monthly Sales Trend",

        "forecast_chart": "Actual vs Forecasted Sales",

        "feature_importance": "Feature Importance",

        "holiday_analysis": "Holiday Impact Analysis",

        "performance_alert": "Store Performance Alert",

        "business_rec": "Business Recommendations",

        "average sales": "Average Monthly Sales",

        "sales by store": "Sales by Store",
        
        "footer":
        "Retail Demand Forecasting Dashboard\nPowered by Random Forest Forecasting Model"
    },

    "中文": {

        "store": "店铺",
        "Weekly_sales": "周销售额",
        
        "Introduction":
        "此仪表板使用随机森林预测模型提供零售销售监控、预测见解和业务建议。",

        "title": "零售需求预测仪表盘",

        "description":
        "该仪表盘提供零售销售监控、销售预测分析以及基于随机森林模型的商业决策建议。",

        "filters": "筛选条件",
        "select_store": "选择门店",
        "select_date": "选择日期范围",
        "minimum_sales": "最低周销售额",

        "kpi": "关键绩效指标",

        "total_sales": "总销售额",
        "forecast_sales": "预测销售额",
        "sales_growth": "销售增长率",
        "forecast_accuracy": "预测准确率",
        "top_store": "最佳门店",

        "top10": "销售额前10门店",
        "bottom10": "销售额后10门店",
        "store_performance": "门店表现",
        "monthly_sales": "月度销售趋势",

        "forecast_chart": "实际销售与预测销售",

        "feature_importance": "特征重要性",

        "holiday_analysis": "节假日影响分析",

        "average sales": "月平均销售额",

        "performance_alert": "门店绩效预警",

        "business_rec": "商业建议",

        "sales by store": "店铺销售额",
        
        "footer":
        "零售需求预测仪表盘\n基于随机森林预测模型"
    }
}
def t(key):
    return translations[language].get(key, key)
# ============================================
# Dashboard Title
# ============================================

st.title(t("title"))

st.markdown(t(
    """Introduction"""
))

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

st.sidebar.header(t("filters"))

# Store filter
selected_store = st.sidebar.multiselect(
    t("select_store"),
    sorted(df['Store'].unique()),
    default=sorted(df['Store'].unique())[:5]
)

# Date filter
min_date = df['Date'].min()
max_date = df['Date'].max()

date_range = st.sidebar.date_input(
    t("select_date"),
    [min_date, max_date]
)

# Sales threshold slider
sales_threshold = st.sidebar.slider(
    t("minimum_sales"),
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

st.subheader(t("kpi"))

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    t("total_sales"),
    f"${total_sales:,.0f}"
)

col2.metric(
    t("forecast_sales"),
    f"${forecast_sales:,.0f}"
)

col3.metric(
    t("sales_growth"),
    f"{sales_growth:.2f}%"
)

col4.metric(
    t("forecast_accuracy"),
    f"{forecast_accuracy:.1f}%"
)

col5.metric(
    t("top_store"),
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

    st.subheader(t("top10"))

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
    title=t("top10"),
    labels={
        "Store": t("store"),
        "Weekly_Sales": t("Weekly_sales")
    }
)

    st.plotly_chart(
        fig_top,
        use_container_width=True
    )

    st.subheader(t("bottom10"))

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
        y='Weekly_Sales'
        title=t("bottom10")
        labels={
        "Store": t("store"),
        "Weekly_Sales": t("Weekly_sales")
    }
    )

    st.plotly_chart(
        fig_bottom,
        use_container_width=True
    )

    st.subheader(t("store_performance"))

    store_sales = (
        filtered_df.groupby('Store')['Weekly_Sales']
        .sum()
        .reset_index()
    )

    fig_bar = px.bar(
        store_sales,
        x='Store',
        y='Weekly_Sales',
        title='Sales by store'
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

    st.subheader(t("monthly_sales"))

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
        title=t("average sales")
    )

    st.plotly_chart(
        fig_month,
        use_container_width=True
    )

# ============================================
# Right Side Main Visualization
# ============================================

with right_col:

    st.subheader(t("forecast_chart"))

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

if language == "中文":

    holiday_df['IsHoliday'] = (
        holiday_df['IsHoliday']
        .replace({
            False: '非节假日',
            True: '节假日'
        })
    )

else:

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
    title=t("holiday_analysis")
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

st.markdown(t("footer"))
