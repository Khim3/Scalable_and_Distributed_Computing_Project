import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pyspark.ml.regression import LinearRegressionModel, DecisionTreeRegressionModel
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.evaluation import RegressionEvaluator
import plotly.express as px
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_spark():
    try:
        sc = SparkContext.getOrCreate()
        spark = SparkSession.builder \
            .appName("Stock Price Prediction") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()
        return spark
    except Exception as e:
        st.error(f"Failed to initialize Spark. Error: {e}")
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
        chart = px.line(data, x='Date', y='Close', title="Stock Closing Prices Over Time")
        st.plotly_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to plot data. Error: {e}")

def load_model(chosen_model):
    try:
        if chosen_model == 'Linear Regression':
            path = 'models/linear_regressor'
            model = LinearRegressionModel.load(path)
        elif chosen_model == 'Decision Tree Regressor':
            path = 'models/decision_tree_regressor'
            model = DecisionTreeRegressionModel.load(path)
        else:
            st.error("Invalid model selected.")
            return None
        st.sidebar.success(f"Model successfully loaded from {path}")
        return model
    except Exception as e:
        st.sidebar.error(f"Failed to load model. Error: {e}")
        return None

def load_data_for_prediction(spark, option, num_days=None):
    try:
        data_test = spark.read.csv('./test.csv', header=True, inferSchema=True)

        if "Date" not in data_test.columns:
            st.error("The 'Date' column is missing in the input data.")
            return None

        data_test = data_test.withColumn("Date", col("Date").cast("date"))
        feature_columns = ["Open", "High", "Low", "Volume"]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        data_test = assembler.transform(data_test)

        data_test = data_test.orderBy(col("Date").asc())
        if option == '1 week':
            data_test = data_test.limit(7)
        elif option == '2 weeks':
            data_test = data_test.limit(14)
        elif option == '1 month':
            data_test = data_test.limit(30)
        elif option == 'Custom' and num_days:
            data_test = data_test.limit(num_days)
        else:
            st.error("Invalid option or number of days not specified.")
        return data_test
    except Exception as e:
        st.error(f"Failed to load data for prediction. Error: {e}")
        return None

def predict(spark, option, chosen_model, num_days=None):
    try:
        data_test = load_data_for_prediction(spark, option, num_days)
        if not data_test:
            return None

        model = load_model(chosen_model)
        if not model:
            return None

        predictions = model.transform(data_test)
        vector_to_array_udf = udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))
        predictions = predictions.withColumn("features_array", vector_to_array_udf(col("features")))
        predictions = predictions.withColumnRenamed("Close", "label")

        evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")
        mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
        rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
        mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
        r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**R-squared (R2):** {r2:.4f}")

        predictions_df = predictions.select("Date", "features_array", "prediction", "label").toPandas()
        predictions_df = predictions_df.sort_values(by="Date")
        predictions_df = predictions_df.drop(columns="features_array")
        return predictions_df
    except Exception as e:
        st.error(f"Failed to predict. Error: {e}")
        return None

def plot_predictions(dataframe, num_days):
    try:
        # Load the original test dataset
        test = pd.read_csv('./test.csv')

        # Ensure the Date column is in the same format for merging
        test["Date"] = pd.to_datetime(test["Date"])
        dataframe["Date"] = pd.to_datetime(dataframe["Date"])

        # Merge predictions with the test data
        merged_data = pd.merge(test, dataframe, on="Date", how="left")

        # Plot actual vs predicted values using Plotly
        fig = go.Figure()

        # Add actual values (Close prices) to the plot
        fig.add_trace(go.Scatter(
            x=merged_data["Date"],
            y=merged_data["Close"],  # Original Close column from test.csv
            mode='lines',
            name='Actual Close Price',
            line=dict(color='purple', width=2)
        ))

        # Add predicted values to the plot
        fig.add_trace(go.Scatter(
            x=merged_data["Date"],
            y=merged_data["prediction"],  # Predicted Close prices
            mode='lines',
            name='Predicted Close Price',
            line=dict(color='red', width=2, dash='dot')
        ))

        # Update layout
        fig.update_layout(
            title=f"Actual vs Predicted Stock Closing Prices for {num_days} Days",
            xaxis_title="Date",
            yaxis_title="Closing Price",
            legend=dict(x=0, y=1),
            template="plotly_white"
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to plot data. Error: {e}")


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
    option = st.sidebar.radio('Choose a forecast period:', ['1 week', '2 weeks', '1 month', 'Custom'])

    num_days = None
    if option == 'Custom':
        num_days = st.sidebar.number_input('Number of days:', min_value=1, value=5)
        st.sidebar.info(f"Forecasting for {num_days} days.")

    if st.sidebar.button('Predict'):
        prediction_df = predict(spark, option, chosen_model, num_days)
        if prediction_df is not None:
            plot_predictions(prediction_df, num_days or option)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
