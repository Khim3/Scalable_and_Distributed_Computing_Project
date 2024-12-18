import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
from pyspark.ml.regression import LinearRegressionModel, DecisionTreeRegressor,  DecisionTreeRegressionModel
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf
import torch
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.evaluation import RegressionEvaluator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    try:
        data = pd.read_csv('./NFLX.csv')
        return data
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        return None
    
def plot_data(data):
    try:
        chart = alt.Chart(data).mark_line().encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Close:Q', title='Closing Price')
        ).properties(title="Stock Closing Prices Over Time")
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to plot data. Error: {e}")


def load_model(chosen_model):
    try:
        if chosen_model == 'Linear Regression':
            path = 'models/linear_regressor'
            model = LinearRegressionModel.load(path)

        elif chosen_model == 'Decision Tree Regressor':
            path = 'models/decision_tree_regressor'
            model = DecisionTreeRegressionModel.load("models/decision_tree_regressor")
            
        elif chosen_model == 'LSTM':
            path = 'models/lstm.pth'
            model = torch.load(path)
            model = model.to(device)
            
        if model:
            st.sidebar.success(f'Model successfully loaded from {path}')
        return model
    except Exception as e:
        st.sidebar.error(f'Failed to load model. Error: {e}')
        return None
    

def load_data_for_prediction(spark, option, num_days=None):
    try:
        # Load the test data
        data_test = spark.read.csv('./test.csv', header=True, inferSchema=True)

        # Define the feature columns
        feature_columns = ["Open", "High", "Low", "Volume"]

        # Create the 'features' column using VectorAssembler
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        data_test = assembler.transform(data_test)

        # Filter or sort based on the option
        if option == '1 day':
            data_test = data_test.orderBy(col("features")).limit(1)
        elif option == '1 week':
            data_test = data_test.orderBy(col("features")).limit(7)
        elif option == '1 month':
            data_test = data_test.orderBy(col("features")).limit(30)
        elif option == 'Custom' and num_days:
            data_test = data_test.orderBy(col("features")).limit(num_days)
        else:
            st.error("Invalid option or number of days not specified.")

        # Return the processed DataFrame
        return data_test
    except Exception as e:
        st.error(f"Failed to load data for prediction. Error: {e}")
        return None
    
def predict(spark, option, chosen_model, num_days=None):
    try:
        # Load and prepare test data
        data_test = load_data_for_prediction(spark, option, num_days)
        if not data_test:
            return

        # Load the model
        model = load_model(chosen_model)
        if not model:
            return

        # Perform prediction
        predictions = model.transform(data_test)

        # Convert DenseVector to list for compatibility
        vector_to_array_udf = udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))
        predictions = predictions.withColumn("features_array", vector_to_array_udf(col("features")))

        # Rename the true value column ('Close') to 'label' for evaluation compatibility
        predictions = predictions.withColumnRenamed("Close", "label")

        # Calculate evaluation metrics
        evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")

        mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
        rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
        mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
        r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

        # Log metrics to Streamlit
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**R-squared (R2):** {r2:.4f}")

        # Select and display predictions
        predictions_df = predictions.select('Date',"features_array", "prediction", "label").toPandas()

        predictions_df = predictions_df.sort_values(by="Date")
        st.write(predictions_df)
    except Exception as e:
        st.error(f"Failed to predict. Error: {e}")
# def prepare_features(data_test):
#     if "features" in data_test.columns:
#         data_test = data_test.drop("features")
#     feature_columns = ["Open", "High", "Low", "Volume"]
#     assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
#     df = assembler.transform(data_test).select("features", col("Close").alias("label"))
#     return df
   
        
def main():
    st.markdown("<h1 style='text-align: center;'>Stock Price Predictions</h1>", unsafe_allow_html=True)
    
    spark = initialize_spark()
    if not spark:
        return
    
    data = load_data()
    if data is not None:
        plot_data(data)
    
    st.sidebar.title('User Input Features')
    chosen_model = st.sidebar.selectbox('Select Algorithm', ('Linear Regression', 'Decision Tree Regressor'))
    option = st.sidebar.radio('Choose a forecast period:', ['1 day', '1 week', '1 month', 'Custom'])
    
    num_days = None
    if option == 'Custom':
        num_days = st.sidebar.number_input('Number of days:', min_value=1, value=5)
        st.sidebar.write(f"Forecasting for {num_days} days.")

    if st.sidebar.button('Predict'):
        prediction =predict(spark, option, chosen_model, num_days)
      
    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
