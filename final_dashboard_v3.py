import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Function to load data
@st.cache_data
def load_data(path_value_1, path_value_2):
    df_value_1 = pd.read_csv(path_value_1, sep='\t')
    df_value_2 = pd.read_csv(path_value_2, sep='\t')
    
    df_value = pd.concat([df_value_1, df_value_2], ignore_index=True)
    df_value = df_value[["OrderID", "EHubName", "OrderDate", "OrdTime", "OrderStatus", "PaymentType", "DeliveryDate", "DeliverySpan", "CouponDiscount", "ShippingAmount", "PayableAmount", "ActualOnlineAmount", "ActualRefundAmount", "MobileType", "RegisteredMobileNo", "City", "State"]]

    # Convert date columns to datetime
    df_value['OrderDate'] = pd.to_datetime(df_value['OrderDate'], format='%d %b %Y')
    df_value['DeliveryDate'] = pd.to_datetime(df_value['DeliveryDate'], format='%d %b %Y %H:%M:%S:%f', errors='coerce')
    df_value['Week-Month'] = df_value['DeliveryDate'].apply(lambda x: f"W{x.isocalendar()[1]}-{x.strftime('%b %Y')}")


    # Preprocessing
    df_value['Week-Month'] = df_value['DeliveryDate'].apply(lambda x: f"W{x.isocalendar()[1]}-{x.strftime('%b %Y')}")
    
    return df_value

# Load the data
path_value_1 = "C:/Users/adm_rvl03/Documents/Nature Basket/Copy of SalesValueReport_May.xls"
path_value_2 = "C:/Users/adm_rvl03/Documents/Nature Basket/Copy of SalesValueReport_June.xls"

try:
    df_value = load_data(path_value_1, path_value_2)
    st.write("Data loaded successfully")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Streamlit app
st.title('Order Status Dashboard')

# Sidebar for user inputs
st.sidebar.header('Options')
x_axis_type = st.sidebar.radio('X-axis Type', ['OrderStatus', 'PaymentType', 'DeliverySpan'])
y_axis_type = st.sidebar.radio('Y-axis Type', ['Absolute Value', 'Percentage'])
metric_type = st.sidebar.radio('Metric Type', ['Count', 'PayableAmount', 'CouponDiscount', 'ShippingAmount', 'ActualRefundAmount'])

# Main part of the dashboard for selecting filters
st.header('Filter Options')
col1, col2, col3 = st.columns(3)
with col1:
    selected_cities = st.multiselect('Select City', df_value['City'].unique())
with col2:
    selected_states = st.multiselect('Select State', df_value['State'].unique())
with col3:
    selected_mobiletypes = st.multiselect('Select Mobile Type', df_value['MobileType'].unique())

col4, col5 = st.columns(2)
with col4:
    min_order_date = df_value['OrderDate'].min()
    max_order_date = df_value['OrderDate'].max()
    order_date_range = st.date_input('Select Order Date Range', [min_order_date, max_order_date])
with col5:
    min_delivery_date = df_value['DeliveryDate'].min()
    max_delivery_date = df_value['DeliveryDate'].max()
    delivery_date_range = st.date_input('Select Delivery Date Range', [min_delivery_date, max_delivery_date])

# Filter data based on selections
if selected_cities:
    df_value = df_value[df_value['City'].isin(selected_cities)]
if selected_states:
    df_value = df_value[df_value['State'].isin(selected_states)]
if selected_mobiletypes:
    df_value = df_value[df_value['MobileType'].isin(selected_mobiletypes)]
    
df_value = df_value[(df_value['OrderDate'] >= pd.to_datetime(order_date_range[0])) & (df_value['OrderDate'] <= pd.to_datetime(order_date_range[1]))]
df_value = df_value[(df_value['DeliveryDate'] >= pd.to_datetime(delivery_date_range[0])) & (df_value['DeliveryDate'] <= pd.to_datetime(delivery_date_range[1]))]

# Calculate average order value
average_order_value = df_value['PayableAmount'].mean()

# Calculate median order value
median_order_value = df_value['PayableAmount'].median()

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Average Order Value (Date)", value=f"₹{average_order_value:,.2f}")
with col2:
    st.metric(label="Median Order Value (Date)", value=f"₹{median_order_value:,.2f}")

# Data processing
try:
    if metric_type == 'Count':
        grouped_data = df_value.groupby(x_axis_type).size().reset_index(name='Count')
        total = grouped_data['Count'].sum()
    else:
        grouped_data = df_value.groupby(x_axis_type)[metric_type].sum().reset_index()
        total = grouped_data[metric_type].sum()

    grouped_data = grouped_data.sort_values(by=grouped_data.columns[1], ascending=False)

    if y_axis_type == 'Percentage':
        grouped_data['Percentage'] = (grouped_data.iloc[:, 1] / total) * 100
        y_column = 'Percentage'
        y_label = '% of ' + ('Orders' if metric_type == 'Count' else metric_type)
    else:
        y_column = 'Count' if metric_type == 'Count' else metric_type
        y_label = y_column

    # Create interactive bar plot using Plotly
    fig = px.bar(grouped_data, x=x_axis_type, y=y_column, title=f'{y_label} by {x_axis_type}', labels={x_axis_type: x_axis_type, y_column: y_label})
    fig.update_layout(
        xaxis_title=x_axis_type,
        yaxis_title=y_label,
        title_x=0.5,
        xaxis=dict(tickfont=dict(color='black'), titlefont=dict(color='black')),
        yaxis=dict(tickfont=dict(color='black'), titlefont=dict(color='black')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black')
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Display the data table
    st.subheader('Data Table')
    st.dataframe(grouped_data)
except Exception as e:
    st.error(f"Error processing data: {e}")

###################################################################################################################
######## Plot Time Series Graph ########
###################################################################################################################

# Date range selector for AOV time series
st.header('Average Order Value Over Time')

# Date range selectors for time series
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        'Start Date',
        df_value['OrderDate'].min(),
        min_value=df_value['OrderDate'].min(),
        max_value=df_value['OrderDate'].max()
    )
with col2:
    end_date = st.date_input(
        'End Date',
        df_value['OrderDate'].max(),
        min_value=df_value['OrderDate'].min(),
        max_value=df_value['OrderDate'].max()
    )

# Filter data for AOV analysis based on selected date range
df_aov = df_value[(df_value['OrderDate'].dt.date >= start_date) & 
                  (df_value['OrderDate'].dt.date <= end_date)]

def calculate_daily_aov(df):
    daily_aov = df.groupby('OrderDate')['PayableAmount'].mean().reset_index()
    daily_aov.columns = ['Date', 'AOV']
    return daily_aov

# Calculate daily AOV
daily_aov = calculate_daily_aov(df_aov)

# Create interactive line plot for AOV over time using Plotly
fig_aov = px.line(daily_aov, x='Date', y='AOV', title='Average Order Value Over Time', labels={'Date': 'Date', 'AOV': 'Average Order Value (₹)'})
fig_aov.update_traces(mode='lines+markers')
fig_aov.update_layout(
    xaxis_title='Date',
    yaxis_title='Average Order Value (₹)',
    title_x=0.5,
    xaxis=dict(tickfont=dict(color='black'), titlefont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'), titlefont=dict(color='black')),
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black')
)

fig_aov.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
fig_aov.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray', tickprefix="₹")

# Display the plot in Streamlit
st.plotly_chart(fig_aov)

# Display additional statistics
st.subheader("Order Value Statistics (Selected Range)")
stats_df = pd.DataFrame({
    'Statistic': ['Minimum', 'Maximum', 'Standard Deviation'],
    'Value': [
        f"₹{df_aov['PayableAmount'].min():,.2f}",
        f"₹{df_aov['PayableAmount'].max():,.2f}",
        f"₹{df_aov['PayableAmount'].std():,.2f}"
    ]
})


st.table(stats_df)

###########################################################################################################
################# Graph of Retention Part #################
###########################################################################################################

# Function to calculate retention
def calculate_retention(df):
    # Get unique Week-Months
    week_months = df['Week-Month'].unique()
    week_months.sort()
    
    retention_data = {}
    
    for i, week_month in enumerate(week_months):
        cohort = df[df['Week-Month'] == week_month]['RegisteredMobileNo'].unique()
        retention_data[week_month] = [100]  # First week is always 100%
        
        for j in range(1, 6):  # We want 6 weeks of retention (including the first week)
            if i + j < len(week_months):
                future_week_month = week_months[i + j]
                retained = df[(df['Week-Month'] == future_week_month) & (df['RegisteredMobileNo'].isin(cohort))]['RegisteredMobileNo'].nunique()
                retention_percentage = (retained / len(cohort)) * 100 if len(cohort) > 0 else 0
            else:
                retention_percentage = None  # No data for this week
            retention_data[week_month].append(retention_percentage)
        
        # Ensure all cohorts have 6 weeks of data
        retention_data[week_month] += [None] * (6 - len(retention_data[week_month]))
    
    return pd.DataFrame(retention_data).T

# Calculate retention
retention_df = calculate_retention(df_value)

# Round to integers, keeping None values as is
retention_percentages = retention_df.applymap(lambda x: round(x) if pd.notnull(x) else None)

# Create a Plotly table
fig = go.Figure(data=[go.Table(
    header=dict(values=['Week-Month'] + [f'Week {i+1}' for i in range(6)],
                fill_color='navy',
                align='left',
                font=dict(color='white', size=12)),
    cells=dict(values=[retention_percentages.index] + [retention_percentages[col] for col in retention_percentages.columns],
               fill_color=[['lightblue']*len(retention_percentages)] + [['white']*len(retention_percentages)]*6,
               align='left',
               font=dict(color='black', size=11),
               format=[None] + ['{:.0f}%' if pd.notnull(x) else '-' for x in retention_percentages[0]]*6))
])

# Update layout
fig.update_layout(
    title='Customer Retention by Week-Month',
    width=800,
    height=400,
)

# Display the table in Streamlit
st.plotly_chart(fig)