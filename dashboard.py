import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_percentage_error
)

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

        "isholiday":"IsHoliday",
        "date": "Date",
        "feature": "Feature",
        "importance": "Importance" ,
        "month": "Month",
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

        "sales forecasting":"Sales Forecasting",
        "predicted_sales": "Predicted Sales",
        "actual_sales": "Actual Sales",
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

        "feature importance": "Feature Importance",

        "holiday impact analysis": "Holiday Impact Analysis",

        "performance_alert": "Store Performance Alert",

        "business_rec": "Business Recommendations",

        "average sales": "Average Monthly Sales",

        "sales by store": "Sales by Store",

        "average sales: holiday vs nonholiday": "Average Sales: Holiday vs Non-Holiday",

        "holiday_analysis": "Holiday Impact Analysis",
        "holiday_type": "Holiday Type",
        "weekly_sales": "Weekly Sales",
        
        "random forest feature importance": "Random Forest Feature Importance",
        "footer":
        "Retail Demand Forecasting Dashboard\nPowered by Random Forest Forecasting Model",

        "normal_performance":
        "🟢 Normal Performance: Sales are within expected range.",

        "warning_performance":
        "🟡 Warning: Sales are below average and should be monitored.",

       "critical_performance":
       "🔴 Critical Alert: Sales are significantly below expected levels.",

       "holiday_recommendation":
       "Increase inventory levels before holiday periods due to higher expected demand.",

      "store_recommendation":
      "Store {store} shows relatively weak performance and may require promotional support.",

     "volatility_recommendation":
     "Sales volatility is currently high. Monitor demand fluctuations carefully.",
    },

    "中文": {

        "isholiday":"是假期",
         "date": "日期",
        "feature": "特征",
        "importance": "重要性" ,
         "month": "月份",
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

        "sales forecasting":"销售预测",
        "predicted_sales": "预测销售额",
        "actual_sales": "实际销售额",
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

        "feature importance": "特征重要性",

        "holiday impact analysis": "节假日影响分析",

        "average sales": "月平均销售额",

        "performance_alert": "门店绩效预警",

        "business_rec": "商业建议",

        "sales by store": "店铺销售额",

        "average sales: holiday vs nonholiday":"平均销售额：假日与非假日",

        "holiday_analysis": "节假日影响分析",
        "holiday_type": "节假日类型",
        "weekly_sales": "周销售额",

        "random forest feature importance": "随机森林特征重要性",
        "footer":
        "零售需求预测仪表盘\n基于随机森林预测模型",
        "normal_performance":
        "🟢 表现正常：销售额处于预期范围内。",

        "warning_performance":
        "🟡 警告：销售额低于平均水平，需要持续关注。",

        "critical_performance":
        "🔴 严重警报：销售额显著低于预期水平。",

        "holiday_recommendation":
        "由于节假日期间需求增加，建议提前提高库存水平。",

       "store_recommendation":
       "门店 {store} 表现相对较弱，建议开展促销活动以提升销售。",

       "volatility_recommendation":
       "当前销售波动较大，建议密切监控需求变化。",
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

    df['Date'] = pd.to_datetime(
        df['Date']
    )
    df['Month'] = df['Date'].dt.month

    return df


df = load_data()

# ============================================
# Train Model - Corrected Version
# ============================================

@st.cache_resource
def train_model(df):

    # --- 增加时间特征 ---
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)

    feature_cols = [
        'Store',
        'Temperature',
        'Fuel_Price',
        'CPI',
        'Unemployment',
        'IsHoliday',
        'rolling_mean_4',
        'rolling_std_4',
        'Year',
        'Month',
        'Week'
    ]

    X = df[feature_cols]
    y = df['Weekly_Sales']

    # --- 时间切分训练/测试 ---
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # --- 随机森林训练 ---
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # --- 测试预测 ---
    y_pred = model.predict(X_test)

    # --- 计算 MAPE ---
    mape = mean_absolute_percentage_error(y_test, y_pred)
    forecast_accuracy = max(0, round((1 - mape) * 100, 1))  # 保证不为负数

    # --- 对整个数据生成预测值 ---
    df['Predicted_Sales'] = model.predict(X)

    # --- 特征重要性 ---
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return df, forecast_accuracy, importance_df, model, mape


# ============================================
# Run Model
# ============================================

df, forecast_accuracy, importance_df, model, mape = train_model(df)

st.write(f"Raw MAPE: {mape*100:.2f}%")
st.write(f"Forecast Accuracy: {forecast_accuracy:.1f}%")

# ============================================
# Sidebar Filters
# ============================================

st.sidebar.header(
    t("filters")
)

selected_store = st.sidebar.multiselect(
    t("select_store"),
    sorted(df['Store'].unique()),
    default=sorted(df['Store'].unique())[:5]
)

min_date = df['Date'].min()

max_date = df['Date'].max()

date_range = st.sidebar.date_input(
    t("select_date"),
    [min_date, max_date]
)

sales_threshold = st.sidebar.slider(
    t("minimum_sales"),
    min_value=0,
    max_value=int(
        df['Weekly_Sales'].max()
    ),
    value=10000
)

# ============================================
# Filter Dataset
# ============================================

filtered_df = df.loc[
    (df['Store'].isin(selected_store)) &
    (df['Date'] >= pd.to_datetime(date_range[0])) &
    (df['Date'] <= pd.to_datetime(date_range[1])) &
    (df['Weekly_Sales'] >= sales_threshold)
].copy()

# 调试用
st.write(
    "Filtered Has Predicted:",
    'Predicted_Sales' in filtered_df.columns
)

# ============================================
# KPI Calculations
# ============================================

total_sales = (
    filtered_df['Weekly_Sales']
    .sum()
)

forecast_sales = (
    filtered_df['Predicted_Sales']
    .sum()
)

sales_growth = (
    filtered_df['Weekly_Sales']
    .pct_change()
    .mean()
    * 100
)

if len(filtered_df) > 0:

    top_store = (
        filtered_df
        .groupby('Store')
        ['Weekly_Sales']
        .sum()
        .idxmax()
    )

else:

    top_store = "N/A"

# ============================================
# KPI Section
# ============================================

st.subheader(
    t("kpi")
)

col1, col2, col3, col4, col5 = st.columns(
    [2, 2, 1, 1, 1]
)

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
# Left Side Charts
# ============================================
left_col, right_col = st.columns(2)
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
        y='Weekly_Sales',
        title=t("bottom10"),
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
        title=t("sales by store"),
        labels={
        "Store": t("store"),
        "Weekly_Sales": t("Weekly_sales")
        }
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
        title=t("average sales"),
        labels={
        "Month": t("month"),
        "Weekly_Sales": t("Weekly_sales")
        }
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

    fig_main = go.Figure()

    fig_main.add_trace(
    go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Weekly_Sales'],
        mode='lines',
        name=t("actual_sales")
        )
    )
   
    fig_main.add_trace(
    go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Predicted_Sales'],
        mode='lines',
        name=t("predicted_sales")
        )
    )
    fig_main.update_layout(
        title=t("sales forecasting"),
        xaxis_title=t("date"),
        yaxis_title=t("Weekly_sales")
    )

    st.plotly_chart(
        fig_main,
        use_container_width=True
    )
# ============================================
# Feature Importance
# ============================================

st.subheader(t("feature importance"))

fig_importance = px.bar(
    importance_df,
    x='Feature',
    y='Importance',
    title=t("random forest feature importance"),
    labels={
    "Feature": t("feature"),
    "Importance": t("importance")
        }
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)

# ============================================
# Holiday Impact Analysis
# ============================================

st.subheader(t("holiday_analysis"))

holiday_df = (
    filtered_df
    .groupby('IsHoliday')['Weekly_Sales']
    .mean()
    .reset_index()
)

# Language Translation
if language == "中文":

    holiday_df['Holiday_Label'] = (
        holiday_df['IsHoliday']
        .replace({
            False: '非节假日',
            True: '节假日'
        })
    )

else:

    holiday_df['Holiday_Label'] = (
        holiday_df['IsHoliday']
        .replace({
            False: 'Non-Holiday',
            True: 'Holiday'
        })
    )

# Holiday Impact Chart
fig_holiday = px.bar(
    holiday_df,
    x='Holiday_Label',
    y='Weekly_Sales',
    color='Holiday_Label',
    title=t("holiday_analysis"),
    labels={
        "Holiday_Label": t("holiday_type"),
        "Weekly_Sales": t("weekly_sales")
    }
)

# Hover Information
fig_holiday.update_traces(
    hovertemplate=
    f"{t('holiday_type')}: %{{x}}<br>"
    f"{t('weekly_sales')}: %{{y:,.0f}}"
    "<extra></extra>"
)

# Layout
fig_holiday.update_layout(
    showlegend=False,
    xaxis_title=t("holiday_type"),
    yaxis_title=t("weekly_sales")
)

st.plotly_chart(
    fig_holiday,
    use_container_width=True
)
# ============================================
# Recommendations Section
# ============================================
st.subheader(t("performance_alert"))

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
    t("normal_performance")
  )

elif ratio >= 0.80:

    st.warning(
    t("warning_performance")
  )

else:

   st.error(
    t("critical_performance")
)
st.subheader(t("business_rec"))

# Recommendation logic
avg_sales = filtered_df['Weekly_Sales'].mean()

holiday_sales = (
    filtered_df[filtered_df['IsHoliday'] == True]
    ['Weekly_Sales']
    .mean()
)

if holiday_sales > avg_sales:
    st.success(
    t("holiday_recommendation")
)

low_store = (
    filtered_df.groupby('Store')['Weekly_Sales']
    .mean()
    .idxmin()
)

st.warning(
    t("store_recommendation").format(
        store=low_store
    )
)

high_variance = filtered_df['rolling_std_4'].mean()

if high_variance > 5000:
    st.info(
    t("volatility_recommendation")
  )

# ============================================
# Footer
# ============================================

st.markdown("---")

st.markdown(t("footer"))
