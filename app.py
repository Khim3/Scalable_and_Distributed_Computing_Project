import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
from pyspark.ml.regression import LinearRegressionModel
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

def initialize_spark():
    try:
        # Get or create a SparkContext
        sc = SparkContext.getOrCreate()

        # Initialize SparkSession using the existing or new SparkContext
        spark = SparkSession.builder \
                .appName("Demo App") \
                .config("spark.some.config.option", "some-value") \
                .getOrCreate()

        return spark
    except Exception as e:
        print(f"Failed to initialize Spark. Error: {e}")
        return None

def load_data():
    data = pd.read_csv('./NFLX.csv')
    return data

def plot_data(data):
    chart = alt.Chart(data).mark_line().encode(
    x=alt.X('Date:T', title='Date'),
    y=alt.Y('Close:Q', title='Closing Price')).properties(
    title="Stock Closing Prices Over Time")

# Display chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

def load_model(chosen_model):
    try:
        if chosen_model == 'Linear Regression':
            path = 'models/linear_regressor'
            model = LinearRegressionModel.load(path)

        else:
            path = 'models/lstm'
            model = Pipeline.load(path)
        # Load the PySpark model
        #st.success(f'Model successfully loaded from {path}')
        return model
    except Exception as e:
        st.sidebar.error(f'Failed to load model. Error: {e}')
        return None

def main():
    data = load_data()
    
    st.markdown("<h1 style='text-align: center;'>Stock Price Predictions</h1>", unsafe_allow_html=True)
    st.sidebar.title('User Input Features') 
    
    # select algorithm
    st.sidebar.info('Welcome to the Stock Price Predictor App!')
  
    
    chosen_model = st.sidebar.selectbox('Select Algorithm', ('Linear Regression', 'SVM Regressor', 'LSTM'))
    load_model(chosen_model)
    st.sidebar.markdown("---")
  
    st.sidebar.subheader('Lựa chọn tác vụ')
    option = st.sidebar.radio('Chọn một tab:', ['1 day', '1 week', '1 month', 'Else'])

    if option == '1 day':
        st.sidebar.write('You choose 1 day')
    elif option == '1 week':
        st.sidebar.write('You choose 1 week')
    elif option == '1 month':
        st.sidebar.write('You choose 1 month')
    else:
        st.sidebar.write('You choose custom input')
        num = st.sidebar.number_input('How many days forecast?', value=5)
        num = int(num)
        
    plot_data(data)
    if st.sidebar.button('Predict') and chosen_model:
        if option == '1 day':
            st.write(f'Predict for 1 day with {chosen_model}')
            
        else: 
            st.write('Quoc bi gay')
        
              

    
if __name__ == '__main__':
    
    initialize_spark()
    main()