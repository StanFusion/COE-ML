import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
import seaborn as sns

#model = joblib.load('best_rfr.pkl')
model = joblib.load('best_rxgboost.pkl')

def predict_price(data):
    prediction = model.predict(data)
    return prediction

def buffer_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    return s

df1 = pd.read_csv("PipelineData.csv")
df1['Successful_diff'] = df1['Quota'] - df1['Successful Bids']
df1['Received'] = df1['Received Bids'] - df1['Quota']

stats_df = df1.groupby('Vehicle Category').agg(
    Diff_Mean=('Successful_diff', 'mean'),
    Diff_Median=('Successful_diff', 'median'),
).reset_index()

stats_df2 = df1.groupby('Vehicle Category').agg(
    Diff_mean=('Received', "mean"),
    Diff_median=('Received', "median"),
    Diff_max=('Received', "max")
).reset_index()

# Streamlit UI
st.title("COE Price Prediction")

if 'page' not in st.session_state:
    st.session_state.page = 'Home'

st.sidebar.title("Navigation")
if st.sidebar.button("Home"):
    st.session_state.page = 'Home'
if st.sidebar.button("Model Prediction"):
    st.session_state.page = 'Model Prediction'
if st.sidebar.button("Data Overview"):
    st.session_state.page = 'Data Overview'
if st.sidebar.button("Extras"):
    st.session_state.page = 'Extras'

# Display content based on the selected page
if st.session_state.page == 'Home':
    st.write("Welcome to the COE Price Prediction App! Use the sidebar to navigate.")
    st.write(" This page contains some information on how to use the predictor.")

    st.write("### For the Field Quota")
    st.write("This refers to the quota up for auction in that particular cycle if unsure")
    st.write("An assumption can be made by dividing the total quota by 6. This total quota is posted on the LTA website and is for 3 months ( 6 bidding execises)")
    st.write("You can also get a more accurate quota on later auctions by calculating how many are left")

    st.write("### For the Field Successful Bids")
    st.write("While we will not have an actual number till the exercise ends. the table below provides some statistics on this field")
    st.table(stats_df)

    st.write("### For the Field Received Bids")
    st.write("Below are some statistics on the difference between the quota and received bids")
    st.table(stats_df2)

    st.write("### For the Field PQP")
    st.write("You can get the required information from this link :https://vrl.lta.gov.sg/lta/vrl/action/ApplicablePQPRates?FUNCTION_ID=F0903006ET")




elif st.session_state.page == 'Model Prediction':
    st.subheader("Model Prediction")
    
    year_input = st.text_input("Enter Year", value="2024") 
    month = st.selectbox('Month', range(1, 13))
    vehicle_category = st.selectbox('Vehicle Category', ['A', 'B', 'C', 'D', 'E']) 
    bidding_type = st.selectbox('Bidding Type', ['1st', '2nd'])
    
    quota = st.number_input('Quota', min_value=0, max_value=2000, value=100)
    successful_bids_value = st.number_input('Successful Bids', min_value=0, max_value=2000, value=10)
    received_bids_value = st.number_input('Received Bids', min_value=0, max_value=2000, value=5)
    pqp_2024 = st.number_input('Prevailing Quota Premium (Average cost in the last 3 months)', min_value=0)

    if st.button('Predict'):
        input_data = pd.DataFrame({
            'Year': [int(year_input)],  
            'Month': [month],
            'Vehicle Category': [vehicle_category],
            'Bidding Type': [bidding_type],
            'Quota': [quota],
            'Successful Bids': [successful_bids_value],
            'Received Bids': [received_bids_value],
            'PQP 2024': [pqp_2024]
        })

        try:
            prediction = predict_price(input_data)
            st.success(f"Predicted COE Price: ${prediction[0]:,.2f}")
        except ValueError as e:
            st.error(f"Error: {str(e)}")

elif st.session_state.page == 'Data Overview':
    st.subheader("Data Overview")
    st.write("Here's a quick overview of the dataset.")
    st.write("Full Datasets will be placed in Extras")

    df = pd.read_csv("PipelineData.csv")
    drop = "Vehicle Category"
    dfcorr = df.copy()
    dfcorr = dfcorr.drop(columns=[drop])

    st.write("Data Preview:", df.head())
    if st.button('Show Dataset Info'):
        st.text(buffer_info(df))

    def plot_histogram(column):
        plt.figure(figsize=(10, 5))
        sns.histplot(df[column], bins=30, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot(plt)
    
    def boxplot(column):
        plt.figure(figsize= (10, 5))
        sns.boxplot(df[column], bins=30, kde=True)
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.ylabel("Frequency")
        st.pyplot(plt)
    

    

    st.write("### Histogram")
    hist_select = st.selectbox("Select Column", 
                               df.select_dtypes
                               (include=['float64', 'int','object']).columns,
                                 index=4,
                                 key="hist_select"
                                 )
    plt.figure(figsize=(5, 3))
    sns.histplot(df[hist_select], bins=30, kde=True)
    key="hist_select"
    st.pyplot(plt)

    st.write("### Box Plot")
    box_select = st.selectbox("Select Column", 
                              df.select_dtypes
                              (include=['float64', 'int','object']).columns, 
                              index=4,
                              key="box_select"
                              )
    plt.figure(figsize=(5, 3))
    sns.boxplot(df[box_select])
    st.pyplot(plt)

    st.write("### Correlation Heatmap")
    corr_matrix = dfcorr.corr()
    plt.figure(figsize=(10, 6))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap")
    st.pyplot(plt)



elif st.session_state.page == 'Extras':
    st.subheader("Extras")
    st.write("Here's a overview of the datasets.")

    dfBefore = pd.read_csv("PipelineData.csv")
    st.write("Below is the data collected from LTA Statistics which contains historical COE Prices")
    st.write("Data Source: https://www.lta.gov.sg/content/ltagov/en/who_we_are/statistics_and_publications/statistics.html")
    st.write("The data can be found under Motor Vehicles in the above link")
    st.dataframe(dfBefore)  

    dfinflate = pd.read_csv("inflate.csv")
    st.write("The inflation value used was collected from the MAS CPI which was taken from https://www.mas.gov.sg/statistics/mas-core-inflation-and-notes-to-selected-cpi-categories")
    st.write("Below is the inflatetion data that was used")
    st.dataframe(dfinflate)

    dfAfter = pd.read_csv("Check.csv")
    st.write("Below is a dataset after accounting for inflation")
    st.dataframe(dfAfter)  