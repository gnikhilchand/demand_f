import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import  mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Load data
@st.cache_data
def load_data():
    transactional_data_1 = pd.read_csv("Transactional_data_retail_01.csv")
    transactional_data_2 = pd.read_csv("Transactional_data_retail_02.csv")
    customer_data = pd.read_csv("CustomerDemographics.csv")
    product_data = pd.read_csv("ProductInfo.csv")
    
    return transactional_data_1, transactional_data_2, customer_data, product_data

# Consolidating transactional data
def consolidate_data(transactions, customers, products):
    # Merge data using SQL-like joins with pandas
    merged_data = pd.merge(transactions, products, on='StockCode')
    merged_data = pd.merge(merged_data, customers, on='Customer ID')
    return merged_data

# Top 10 stock codes by quantity sold
def top_10_by_quantity(merged_data):
    top_10 = merged_data.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(10)
    return top_10

# Top 10 products by revenue
def top_10_by_revenue(merged_data):
    merged_data['Revenue'] = merged_data['Quantity'] * merged_data['Price']
    top_10 = merged_data.groupby('StockCode')['Revenue'].sum().sort_values(ascending=False).head(10)
    return top_10

# EDA - summary stats
def eda_summary(merged_data):
    st.write("### Summary Statistics")
    st.write(merged_data.describe())

# ACF and PACF plots
def acf_pacf_plots(data, lags=40):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(data, lags=lags, ax=ax[0])
    plot_pacf(data, lags=lags, ax=ax[1])
    st.pyplot(fig)

# Train ARIMA model
def train_arima(data, order):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

# Train Prophet model
def train_prophet(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])
    df_prophet = df[['InvoiceDate', 'Quantity']].rename(columns={'InvoiceDate': 'ds', 'Quantity': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    return model

# Forecasting
def forecast_prophet(model, periods=15):
    future = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)
    return forecast

# Error histogram
# def error_histogram(actual, predicted, dataset_type="Train"):
#     actual = np.array(actual)

#     if predicted.ndim == 2:
#         predicted = predicted[:, 0]

#     predicted = np.array(predicted)

#     if len(actual) != len(predicted):
#         raise ValueError("Both arrays must have the same length")

#     if np.issubdtype(actual.dtype, np.datetime64):
#         actual = actual.astype('float64')
#     if np.issubdtype(predicted.dtype, np.datetime64):
#         predicted = predicted.astype('float64')

#     errors = actual - predicted
#     errors = errors[~np.isnan(errors)]

#     fig = plt.figure()
#     plt.hist(errors, bins=20)
#     plt.title(f'{dataset_type} Error Histogram')
#     st.pyplot(fig)

# Main app logic
def main():
    st.title("Demand Forecasting App")

    # Load data
    trans_data_1, trans_data_2, customer_data, product_data = load_data()
    
    # Concatenate the two transactional datasets
    transactional_data = pd.concat([trans_data_1, trans_data_2])
    
    # Consolidate data
    merged_data = consolidate_data(transactional_data, customer_data, product_data)
    
    # Sidebar options
    st.sidebar.header("Choose Analysis")
    option = st.sidebar.selectbox("Select analysis type", ["EDA", "Top 10 Analysis", "Time Series Forecasting"])
    
    if option == "EDA":
        eda_summary(merged_data)
        
        # Visualizations for summary
        st.write("### Quantity Distribution")
        fig = px.histogram(merged_data, x='Quantity')
        st.plotly_chart(fig)
        
        st.write("### Revenue Distribution")
        fig = px.histogram(merged_data, x='Price')
        st.plotly_chart(fig)

    elif option == "Top 10 Analysis":
        st.write("### Top 10 Stock Codes by Quantity Sold")
        top_10_quantity = top_10_by_quantity(merged_data)
        st.bar_chart(top_10_quantity)
        
        st.write("### Top 10 Products by Revenue")
        top_10_revenue = top_10_by_revenue(merged_data)
        st.bar_chart(top_10_revenue)
    
    elif option == "Time Series Forecasting":
        # Choose stock code
        top_10_quantity = top_10_by_quantity(merged_data).index.tolist()
        stock_code = st.sidebar.selectbox("Select Stock Code", top_10_quantity)
        
        # Filter data for selected stock code
        stock_data = merged_data[merged_data['StockCode'] == stock_code]
        
        # Time Series Plot
        st.write(f"### Time Series for Stock Code: {stock_code}")
        fig = px.line(stock_data, x='InvoiceDate', y='Quantity')
        st.plotly_chart(fig)
        
        # ACF and PACF
        st.write("### ACF and PACF Plots")
        acf_pacf_plots(stock_data['Quantity'])
        
        # Forecasting with ARIMA or Prophet
        model_type = st.sidebar.radio("Select Model", ["ARIMA", "Prophet"])
        if model_type == "ARIMA":
            arima_order = st.sidebar.text_input("Enter ARIMA Order (p, d, q)", "1,1,1")
            arima_order = tuple(map(int, arima_order.split(',')))
            arima_model = train_arima(stock_data['Quantity'], arima_order)
            forecast = arima_model.forecast(steps=15)
            st.write("### Forecast for Next 15 Weeks")
            st.line_chart(forecast)
        
        elif model_type == "Prophet":
            prophet_model = train_prophet(stock_data)
            forecast = forecast_prophet(prophet_model)
            st.write("### Forecast for Next 15 Weeks")
            st.line_chart(forecast['yhat'])
        
        # Error histogram
        # st.write("### Error Histogram")
        # error_histogram(stock_data['Quantity'], forecast)

if __name__ == "__main__":
    main()
