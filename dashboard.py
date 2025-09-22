# dashboard.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("customer_behavior_timeseries.csv")

# Convert month to datetime
df['month'] = pd.to_datetime(df['month'])

# Add active column
df['active'] = df['is_churned'].apply(lambda x: 0 if x==1 else 1)
df['monthly_revenue'] = df['monthly_spent']

# -----------------------
# Sidebar Filters
# -----------------------
st.sidebar.header("Filters")
start_date = st.sidebar.date_input("Start Date", df['month'].min())
end_date = st.sidebar.date_input("End Date", df['month'].max())
gender_filter = st.sidebar.multiselect("Gender", df['gender'].unique(), default=df['gender'].unique())

# Apply filters
filtered_df = df[(df['month'] >= pd.to_datetime(start_date)) &
                 (df['month'] <= pd.to_datetime(end_date)) &
                 (df['gender'].isin(gender_filter))]

# -----------------------
# KPI Section
# -----------------------
st.title("📊 Customer Behavior Dashboard")

total_revenue = filtered_df['monthly_revenue'].sum()
total_visits = filtered_df['visits'].sum()
total_purchases = filtered_df['purchases'].sum()
active_customers = filtered_df.groupby('customer_id')['active'].max().sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Revenue", f"${total_revenue:,.0f}")
col2.metric("👥 Active Customers", f"{active_customers}")
col3.metric("🖱 Total Visits", f"{total_visits}")
col4.metric("🛒 Total Purchases", f"{total_purchases}")

# -----------------------
# Aggregated Monthly Summary
# -----------------------
monthly_summary = filtered_df.groupby('month').agg({
    'visits':'sum',
    'purchases':'sum',
    'monthly_revenue':'sum',
    'customer_id':'nunique'
}).rename(columns={'customer_id':'active_customers'}).reset_index()

monthly_summary['revenue_growth'] = monthly_summary['monthly_revenue'].pct_change()*100
monthly_summary['visits_growth'] = monthly_summary['visits'].pct_change()*100
monthly_summary['purchases_growth'] = monthly_summary['purchases'].pct_change()*100

# -----------------------
# Interactive Plots
# -----------------------

st.subheader("📈 Monthly Revenue Trend")
fig_revenue = px.line(monthly_summary, x='month', y='monthly_revenue', markers=True,
                      title="Monthly Revenue Trend")
st.plotly_chart(fig_revenue, use_container_width=True)

st.subheader("📊 Monthly Website Visits")
fig_visits = px.bar(monthly_summary, x='month', y='visits', title="Monthly Visits")
st.plotly_chart(fig_visits, use_container_width=True)

st.subheader("🛒 Monthly Purchases")
fig_purchases = px.bar(monthly_summary, x='month', y='purchases', title="Monthly Purchases")
st.plotly_chart(fig_purchases, use_container_width=True)

st.subheader("📉 Churn Status (Latest Month)")
latest_month = filtered_df['month'].max()
latest_data = filtered_df[filtered_df['month']==latest_month]
fig_churn = px.pie(latest_data, names='is_churned', 
                   title=f"Customer Churn Status ({latest_month.strftime('%b %Y')})",
                   labels={'is_churned':'Churned'})
st.plotly_chart(fig_churn, use_container_width=True)

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

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("Dashboard created with ❤️ using **Streamlit** and **Plotly**.")
