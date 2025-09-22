# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------
# Page config (FIRST Streamlit command!)
# -----------------------
st.set_page_config(
    page_title="Advanced Customer Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)
# -----------------------
# Hide header & footer
# -----------------------
st.markdown(
    """
    <style>
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Sidebar: Theme selector
# -----------------------
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("customer_behavior_timeseries.csv")
df['month'] = pd.to_datetime(df['month'])
df['active'] = df['is_churned'].apply(lambda x: 0 if x == 1 else 1)
df['monthly_revenue'] = df['monthly_spent']

# -----------------------
# Sidebar Filters
# -----------------------
st.sidebar.header("Filters")
start_date = st.sidebar.date_input("Start Date", df['month'].min())
end_date = st.sidebar.date_input("End Date", df['month'].max())
gender_filter = st.sidebar.multiselect("Gender", df['gender'].unique(), default=df['gender'].unique())

# Apply filters
filtered_df = df[
    (df['month'] >= pd.to_datetime(start_date)) &
    (df['month'] <= pd.to_datetime(end_date)) &
    (df['gender'].isin(gender_filter))
]

# -----------------------
# Monthly summary and growth %
# -----------------------
monthly_summary = filtered_df.groupby('month').agg({
    'visits': 'sum',
    'purchases': 'sum',
    'monthly_revenue': 'sum',
    'customer_id': 'nunique'
}).rename(columns={'customer_id': 'active_customers'}).reset_index()

monthly_summary['revenue_growth'] = monthly_summary['monthly_revenue'].pct_change().fillna(0) * 100
monthly_summary['visits_growth'] = monthly_summary['visits'].pct_change().fillna(0) * 100
monthly_summary['purchases_growth'] = monthly_summary['purchases'].pct_change().fillna(0) * 100

latest = monthly_summary.iloc[-1]

# -----------------------
# KPI Cards
# -----------------------
st.title("📊 Advanced Customer Behavior Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("💰 Revenue", f"${latest['monthly_revenue']:,.0f}", f"{latest['revenue_growth']:.1f}%")
col2.metric("👥 Active Customers", f"{latest['active_customers']}", f"{filtered_df['active'].mean()*100:.1f}% Active")
col3.metric("🛒 Purchases", f"{latest['purchases']}", f"{latest['purchases_growth']:.1f}%")

# -----------------------
# Hoverable Monthly Trends
# -----------------------
st.subheader("📈 Monthly Trends (Hover for Details)")
fig_trends = go.Figure()
template_style = "plotly_dark" if theme == "Dark" else "plotly_white"

fig_trends.add_trace(go.Scatter(x=monthly_summary['month'], y=monthly_summary['monthly_revenue'],
                                mode='lines+markers', name='Revenue'))
fig_trends.add_trace(go.Scatter(x=monthly_summary['month'], y=monthly_summary['visits'],
                                mode='lines+markers', name='Visits'))
fig_trends.add_trace(go.Scatter(x=monthly_summary['month'], y=monthly_summary['purchases'],
                                mode='lines+markers', name='Purchases'))

fig_trends.update_layout(
    hovermode="x unified",
    title="Monthly Revenue, Visits, Purchases",
    template=template_style
)
st.plotly_chart(fig_trends, use_container_width=True)

# -----------------------
# Heatmap: Visits vs Purchases
# -----------------------
st.subheader("🔥 Heatmap: Visits vs Purchases")
heatmap_data = filtered_df.pivot_table(values='customer_id', index='visits', columns='purchases', aggfunc='count', fill_value=0)

fig, ax = plt.subplots(figsize=(8,6))
sns.set_theme(style="darkgrid" if theme=="Dark" else "whitegrid")
sns.heatmap(heatmap_data, cmap="Blues", ax=ax)
st.pyplot(fig)

# -----------------------
# Cohort Analysis for Churn
# -----------------------
st.subheader("📊 Cohort Analysis: Churn over Time")
filtered_df['cohort_month'] = filtered_df.groupby('customer_id')['month'].transform('min')
cohort_data = filtered_df.groupby(['cohort_month','month']).agg({
    'customer_id':'nunique',
    'is_churned':'mean'
}).reset_index()

# Fixed month calculation (no np.timedelta64('M'))
cohort_data['period'] = (
    (cohort_data['month'].dt.year - cohort_data['cohort_month'].dt.year) * 12 +
    (cohort_data['month'].dt.month - cohort_data['cohort_month'].dt.month)
)

cohort_pivot = cohort_data.pivot(index='cohort_month', columns='period', values='is_churned').fillna(0)

fig2, ax2 = plt.subplots(figsize=(10,6))
sns.heatmap(cohort_pivot, cmap="Reds", annot=True, fmt=".1f", ax=ax2)
ax2.set_title("Cohort Analysis: Churn Rate by Cohort and Period")
st.pyplot(fig2)

# -----------------------
# Download Filtered Data
# -----------------------
st.subheader("💾 Download Filtered Data")
st.download_button(
    "Download CSV",
    filtered_df.to_csv(index=False),
    file_name='filtered_customer_data.csv',
    mime='text/csv'
)

st.markdown("---")
