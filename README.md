# Stock Price Forecasting System using Apache Spark

## Introduction

This repository contains the implementation of a stock price forecasting system for Netflix, designed to assist in making data-driven investment decisions. By leveraging Apache Spark for distributed computing and advanced machine learning models, the system analyzes historical stock data and predicts future prices. 

The project incorporates linear regression, decision tree regression, and LSTM models to benchmark performance and identify the best model for forecasting. A Streamlit-based user interface (UI) has been developed to load pre-trained models, test them on provided test files, and visualize the results with actual vs. predicted plots.

## Team Members 

| Order |         Name          |     ID      |                  Email                  |                       Github account                        |                                                       
| :---: | :-------------------: | :---------: |:---------------------------------------:| :---------------------------------------------------------: | 
|   1   | Nguyen Tri Tin | ITDSIU21123 |  ITDSIU21123@student.hcmiu.edu.vn | [kumaekiku](https://github.com/Son-SDT) | 
|   2   | Nguyen Nhat Khiem | ITDSIU21091 |  ITDSIU21091@student.hcmiu.edu.vn | [Khim3](https://github.com/Khim3) | 
|   3   | Nguyen Tran Viet Khoi | ITCSIU21081 |  ITCSIU21081@student.hcmiu.edu.vn | [tpSpace](https://github.com/tpSpace) |

## Getting Started
### Prerequisites
- Apache Spark (version 3.5.3)
- Python3 (version 3.6 or higher)
- Jupyter Notebook
- Necessary Python libraries (listed in `requirements.txt`)
- Streamlit (version 1.41.1)
- Java (version 8 or higher)
### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Khim3/Scalable_and_Distributed_Computing_Project
   ```
2. Install dependencies:
   ```bash
    pip install -r requirements.txt
    ```
3. Run the application
   ```bash
    streamlit run app.py
   ```
    or access the web application at: Stock Price Forecasting System [http://34.126.166.36:8501/]

Or you could run it on Docker (make sure your internet is strong enough for downloading Python packages)

1. Build docker images:
   ```
   docker build -t app .
   ```
2. Run images by docker compose
   ```
   docker compose up -d
   ```

## Tools Used
| **Tool**                     | **Purpose**                                                                                       |
|------------------------------|---------------------------------------------------------------------------------------------------|
| **Streamlit**                | Build the interactive web application for displaying stock price forecasting results for users to interact.            |
| **Python**                   | Main programming language for implementing the forecasting logic and integrating the tools.       |
| **PySpark**                  | Handle large-scale data processing and perform distributed data analysis and transformations.     |
| **Pandas**                   | Turn Spark dataframe to data type that plotting libraries could easily handle.                          |
| **Plotly**                   | Visualize stock price trends, historical data, and predictions in a user-friendly format.         |
| **scikit-learn**             | Provide LSTM model 's evaluation metrics            |
| **PyTorch**                  | Implement deep learning models for more advanced forecasting (LTSM) |
| **GitHub**                   | Host the project codebase and facilitate version control and collaboration.                       |
| **Docker**                   | For ease of deployment on cloud server or others machine                                          |
| **GCP**                      | We use Google Cloud Platform to deployment our application                                        |

## Features
- **Data Investigation and Preprocessing:** Handling missing values, adjusting for stock splits/dividends, and aligning timestamps.
- **Predictive Modeling:** Implementing and comparing linear regression, decision tree regression, and LSTM models.
- **Model Evaluation:** Using metrics such as MAE, RMSE, and directional accuracy to assess model performance.
- **Streamlit UI:** A web-based interface for testing models with test files and visualizing predictions.

## Deployment
We use Docker to dockerize our application
- **Portability:** Docker containers can run on any system with Docker installed.
- **Consistency:** Ensures the application works seamlessly regardless of the host environment.
- **Scalability:** Easily scale containers in a cloud environment like GCP.
- **Isolation:** Applications run in separate containers, avoiding dependency conflicts.
  
And use GCP for deploy for public use.
- **Global Reach:** Provides low-latency access via Google's global network.
- **Integration:** Seamlessly integrates with other Google services (e.g., Cloud Storage, BigQuery).
- **Cost Optimization:** Pay-as-you-go model ensures efficient resource usage.

## Project Structure

- ./models/: Contains pre-trained models for linear regression, decision tree regression, and LSTM.
- ./notebooks/: Jupyter notebooks for data preprocessing, model training, and evaluation.
- ./data/: Contains historical stock data for Netflix and test files for model evaluation.
- ./app.py: Streamlit-based web application for loading models, testing on test files, and visualizing predictions.
- ./requirements.txt: List of Python libraries required for the project.
- ./lsmt.py: Implementation of the LSTM model architecture for stock price forecasting.
- ./scaler.pkl: Pre-trained MinMaxScaler for normalizing input data for LSTM model.
- ./README.md: Project documentation with an overview, team members, installation instructions, and features.

## Accomplishments
- **Data Preprocessing:** Cleaned and preprocessed historical stock data to ensure consistency and accuracy in forecasting.
- **Model Implementation:** Developed linear regression, decision tree regression, and LSTM models for stock price forecasting.
- **Model Evaluation:** Evaluated model performance using metrics such as MAE, RMSE, and directional accuracy.
- **Streamlit UI:** Created an interactive web application for users to load models, test on provided files, and visualize predictions.
- **Documentation:** Wrote detailed documentation for the project, including an overview, team members, installation instructions, and features.

## Limitations
- **Data Quality:** The accuracy of stock price forecasting is highly dependent on the quality of historical data and the assumptions made during preprocessing.
- **Model Selection:** The choice of forecasting model can significantly impact the accuracy of predictions, and different models may perform better under different conditions.
- **Evaluation Metrics:** While metrics such as MAE and RMSE provide insights into model performance, they may not capture all aspects of forecasting accuracy.
- **User Interface:** The Streamlit-based UI provides basic functionality for loading models and testing on test files but could be further enhanced with additional features.
- **Data Limitations:** The forecasting system is limited to historical stock data for Netflix (only for 5 years) and may not generalize well to other stocks or financial instruments.

## Future Work

- **Model Tuning:** Experiment with hyperparameter tuning and feature engineering to improve the accuracy of forecasting models.
- **Ensemble Methods:** Explore the use of ensemble methods such as bagging and boosting to combine multiple models for better predictions.
- **Additional Features:** Enhance the Streamlit UI with additional features such as real-time data updates, custom model training, and more advanced visualizations.
- **Deployment:** Deploy the forecasting system to a cloud platform such as AWS or GCP to make it accessible to a wider audience.
- **Test with more models**: Test with more models to find the best model for forecasting stock price.

## Acknowledgements
- We would like to thank our instructor, Dr. Mai Hoang Bao An, for his guidance and support throughout the project.
- We would also like to thank our classmates for their feedback and suggestions during the development of the forecasting system.
- Finally, we would like to thank the open-source community for providing tools and libraries that made this project possible.
- We would like to thank the authors of the following resources for their valuable insights and contributions to the field of stock price forecasting.
## Contribution
This is a school project, set up and maintained merely by our team, so we are not looking for contributions. However, if you have any suggestions or feedback, feel free to create an issue or contact us directly. We look forward to hearing from you!

## Project structure
```
📦 
├─ .github
│  └─ workflows
│     └─ cicd.yml
├─ .gitignore
├─ Dockerfile
├─ README.md
├─ __pycache__
│  └─ lstm.cpython-310.pyc
├─ app.py
├─ data
│  ├─ NFLX.csv
│  └─ test.csv
├─ docker-compose.yaml
├─ lstm.py
├─ models
│  ├─ decision_tree_regressor
│  │  ├─ data
│  │  │  ├─ ._SUCCESS.crc
│  │  │  ├─ .part-00000-88cf27f8-8b9a-46f7-bbe6-9a26b8bfdc0c-c000.snappy.parquet.crc
│  │  │  ├─ _SUCCESS
│  │  │  └─ part-00000-88cf27f8-8b9a-46f7-bbe6-9a26b8bfdc0c-c000.snappy.parquet
│  │  └─ metadata
│  │     ├─ ._SUCCESS.crc
│  │     ├─ .part-00000.crc
│  │     ├─ _SUCCESS
│  │     └─ part-00000
│  ├─ linear_regressor
│  │  ├─ data
│  │  │  ├─ ._SUCCESS.crc
│  │  │  ├─ .part-00000-18ef8e6c-a22c-4aba-9132-5801c861b98b-c000.snappy.parquet.crc
│  │  │  ├─ _SUCCESS
│  │  │  └─ part-00000-18ef8e6c-a22c-4aba-9132-5801c861b98b-c000.snappy.parquet
│  │  └─ metadata
│  │     ├─ ._SUCCESS.crc
│  │     ├─ .part-00000.crc
│  │     ├─ _SUCCESS
│  │     └─ part-00000
│  └─ lstm.pth
├─ nginx
│  └─ default.conf
├─ notebook.ipynb
├─ notebooks
│  ├─ DT_Regressor.ipynb
│  ├─ Linear_Regressor.ipynb
│  ├─ data_preprocess.ipynb
│  ├─ ltsm.ipynb
│  └─ test.ipynb
├─ pyproject.toml
├─ requirements.txt
├─ scaler.pkl
└─ wrapper.py
```
©generated by [Project Tree Generator](https://woochanleee.github.io/project-tree-generator)

## References
1. [Stock Price Prediction Using Machine Learning and Deep Learning Techniques](https://www.researchgate.net/publication/344091013_Stock_Price_Prediction_Using_Machine_Learning_and_Deep_Learning_Techniques)
2. [Stock Price Prediction Using Notebook on Kaggle - Armaan Seth](https://www.kaggle.com/code/armaanseth6702/stockpriceprediction/notebook)
3. [Stock Price Prediction Using LSTM - 034adarsh](https://github.com/034adarsh/Stock-Price-Prediction-Using-LSTM/blob/main/LSTM_model.ipynb)
4. [Stock-Prediction-with-Python-project  -- Hoang Xuan Quoc](
   https://stock-prediction-with-python-project-quocchienduc.streamlit.app/)
5. [Source data on Kaggle](https://www.kaggle.com/datasets/jainilcoder/netflix-stock-price-prediction)
   

