import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objs as go
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

st.title('Time Series Forecasting Explorer')

# Introduction
st.write("""
Time series forecasting is the use of a model to predict future values based on previously observed values.
This app provides information about time series forecasting, different methods, and models.
""")
# Explanation of Forecasting in the LSTM App

st.header("How to Use the Web Application")

st.markdown("""
1. **Upload CSV File**:
   - Click on the "Choose a CSV file" button.
   - Select and upload a CSV file from your computer. Ensure that the CSV file has 'date' and 'value' columns.

2. **Dataset Preview**:
   - After uploading, the app will display a preview of the dataset.
   - Verify that the dataset has been loaded correctly and contains the necessary columns.

3. **Check Dataset Information**:
   - Look for the displayed number of rows in the dataset to ensure data has been loaded correctly.

4. **Select Model Configuration**:
   - From the dropdown menu, select an LSTM model configuration:
     - **Model 1 (32 Units)**: Suitable for faster training with smaller datasets.
     - **Model 2 (56 Units)**: Balances between training time and model complexity.
     - **Model 3 (64 Units)**: Provides a higher capacity model for capturing patterns.
     - **Model 4 (80 Units)**: Offers more complexity and is suitable for larger datasets.
     - **Model 5 (128 Units)**: The most complex model, ideal for capturing intricate patterns but requires more training time.
   - Optionally, you can choose "All Models" to train and evaluate all model configurations.

5. **Set Forecast Days**:
   - Use the number input widget to select the number of days you want to forecast.

6. **Start the Task**:
   - Click on the "Start Tasks" button to initiate the data preprocessing, model training, and evaluation process.

7. **View Basic Plot**:
   - A plot showing the value over time will be displayed. You can hover over the plot to see the values.

8. **Model Training and Loss Plot**:
   - The app will train the selected LSTM model and display a plot of the training loss over epochs.

9. **View Predictions**:
    - After training, the app will display the predicted values compared to the actual values.
    - You can hover over the plots to see the predicted and actual values.

10. **Future Forecast Plot**:
    - A plot showing the forecasted values for the selected number of days will be displayed.
    - Hover over the plot to see the forecasted values.

11. **Evaluation Metrics**:
    - The app will display evaluation metrics such as RMSE, MSE, MAE, and R-squared for the selected model.
12. **Compare Multiple Models (Optional)**:
    - If "All Models" was selected, the app will train and evaluate each model configuration sequentially and display their respective results.

13. **Review Last 30 Values and Forecasted Values**:
    - The app will display a table of the last 30 values and the forecasted values for easy comparison.

14. **Select Another Model (Optional)**:
    - If desired, select a different model configuration from the dropdown and repeat the process.

15. **Exit the Application**:
    - Once done, you can close the web application.
""")
# Default Dataset Download Link
st.subheader('Default Dataset')
st.write("""
You can download this default dataset to use with this app.
- [Default Dataset CSV](https://drive.google.com/file/d/1Q2mQrPuU_T_mIuTGTYoqTXPiSpe9W57L/view?usp=sharing)
""")


# Interactive Graph - Sample Time Series Data
st.subheader('Sample Time Series Data')
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2020, 12, 31)
idx = pd.date_range(start=start_date, end=end_date, freq='D')
data = np.random.randn(len(idx))
ts_data = pd.Series(data, index=idx)
fig_ts_data = go.Figure()
fig_ts_data.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines', name='Time Series Data'))
fig_ts_data.update_layout(title='This is how a time series dataset looks like in a graph ', xaxis_title='Date', yaxis_title='Value')
st.plotly_chart(fig_ts_data)

