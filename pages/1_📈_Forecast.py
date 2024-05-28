import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Let's forecast with LSTM Model")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file (Note: Make sure date and value columns are present in the dataset)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())
    st.write(df.tail())
    
    # Display the number of rows in the dataset
    st.write(f"Number of rows in the dataset: {len(df)}")
    
    # Check if 'date' and 'value' columns are present
    if 'date' not in df.columns or 'value' not in df.columns:
        st.warning("Hey, I think your dataset does not have the 'date' and/or 'value' column(s). Please upload a CSV file with these columns.")
    else:
        # Attempt to convert 'date' column to datetime
        try:
            df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        except Exception as e:
            st.warning(f"The 'date' column could not be converted to datetime. Please make sure the 'date' column is in the correct format. Error: {e}")
        
        # Check if 'value' column contains numerical data type
        if not np.issubdtype(df['value'].dtype, np.number):
            st.warning("The 'value' column should contain numerical data type. Please make sure the 'value' column is in the correct format.")
        else:
            model_options = ["Model 1(32 Units)", "Model 2(56 Units)", "Model 3(64 Units)", "Model 4(80 Units)", "Model 5(128 Units)", "All Models"]
            model_choice = st.selectbox("Select an LSTM model configuration (Models with 80 and 128 units will take longer time to train):", model_options)

            # Select the number of days to forecast
            forecast_days = st.number_input("Select number of days to forecast:", min_value=1, max_value=100, value=30)

            if st.button("Start Tasks"):
                df = df[['date', 'value']]
                
                # Data preprocessing
                df.dropna(inplace=True)
                df.drop_duplicates(inplace=True)
                z_scores = stats.zscore(df['value'])
                df = df[(z_scores < 3)]    
                df.reset_index(drop=True, inplace=True)
                
                st.subheader('Basic Plot')
                fig = px.line(df, x='date', y='value', title='Value Over Time')
                fig.update_layout(plot_bgcolor='white')
                st.plotly_chart(fig)
                
                # Extract the series to be normalized
                ds = df["value"]

                # Normalize the data
                scaler = MinMaxScaler(feature_range=(0, 1))
                ds_scaled = scaler.fit_transform(np.array(ds).reshape(-1, 1))

                # Define train and test sizes
                train_size = int(len(ds_scaled) * 0.7)
                test_size = len(ds_scaled) - train_size

                # Split data into train and test sets
                ds_train, ds_test = ds_scaled[0:train_size, :], ds_scaled[train_size:len(ds_scaled), :1]

                # Create dataset in time series for LSTM model
                def create_ds(dataset, step):
                    X, Y = [], []
                    for i in range(len(dataset) - step - 1):
                        a = dataset[i:(i + step), 0]
                        X.append(a)
                        Y.append(dataset[i + step, 0])
                    return np.array(X), np.array(Y)

                # Taking 100 days price as one record for training
                time_stamp = 100
                X_train, y_train = create_ds(ds_train, time_stamp)
                X_test, y_test = create_ds(ds_test, time_stamp)

                # Reshape data to fit into LSTM model
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                with st.spinner('Please wait, the model training has started...'):
                    # Define LSTM models configurations
                    def get_model(choice):
                        model = Sequential()
                        if choice == "Model 1(32 Units)":
                            model.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                            model.add(LSTM(units=32, return_sequences=True))
                            model.add(LSTM(units=32))
                        elif choice == "Model 2(56 Units)":
                            model.add(LSTM(units=56, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                            model.add(LSTM(units=56, return_sequences=True))
                            model.add(LSTM(units=56))
                        elif choice == "Model 3(64 Units)":
                            model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                            model.add(LSTM(units=64, return_sequences=True))
                            model.add(LSTM(units=64))
                        elif choice == "Model 4(80 Units)":
                            model.add(LSTM(units=80, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                            model.add(LSTM(units=80, return_sequences=True))
                            model.add(LSTM(units=80))
                        elif choice == "Model 5(128 Units)":
                            model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                            model.add(LSTM(units=128, return_sequences=True))
                            model.add(LSTM(units=128))
                        model.add(Dense(units=1, activation='linear'))
                        return model

                    # Train and evaluate models
                    def train_and_evaluate_model(model_choice, X_train, y_train, X_test, y_test):
                        model = get_model(model_choice)
                        model.compile(loss='mean_squared_error', optimizer='adam')
                        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)
                        
                        # Training loss plot
                        st.subheader('Training Loss')
                        fig = px.line(pd.DataFrame({'loss': history.history['loss']}), y='loss', title='Model Loss')
                        fig.update_layout(plot_bgcolor='white')
                        st.plotly_chart(fig)
                        
                        # Predict
                        train_predict = model.predict(X_train)
                        test_predict = model.predict(X_test)

                        # Inverse transform to get actual value
                        train_predict = scaler.inverse_transform(train_predict)
                        test_predict = scaler.inverse_transform(test_predict)

                        # Plot predictions
                        st.subheader(f'Prediction of the model over actual dataset {model_choice}')
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=np.arange(len(ds_scaled)), y=scaler.inverse_transform(ds_scaled).flatten(), mode='lines', name='Actual'))
                        fig.add_trace(go.Scatter(x=np.arange(len(train_predict)), y=train_predict.flatten(), mode='lines', name='Train Predict'))
                        #fig.add_trace(go.Scatter(x=np.arange(len(train_predict), len(train_predict) + len(test_predict)), y=test_predict.flatten(), mode='lines', name='Test Predict'))
                        fig.update_layout(plot_bgcolor='white')
                        st.plotly_chart(fig)

                        # Future prediction
                        fut_inp = ds_scaled[len(ds_scaled) - 100:]
                        fut_inp = fut_inp.reshape(1, -1)
                        tmp_inp = list(fut_inp)
                        tmp_inp = tmp_inp[0].tolist()

                        lst_output = []
                        n_steps = 100
                        i = 0
                        while(i < forecast_days):
                            if(len(tmp_inp) > 100):
                                fut_inp = np.array(tmp_inp[1:])
                                fut_inp = fut_inp.reshape(1, -1)
                                fut_inp = fut_inp.reshape((1, n_steps, 1))
                                yhat = model.predict(fut_inp, verbose='auto')
                                tmp_inp.extend(yhat[0].tolist())
                                tmp_inp = tmp_inp[1:]
                                lst_output.extend(yhat.tolist())
                                i += 1
                            else:
                                fut_inp = fut_inp.reshape((1, n_steps, 1))
                                yhat = model.predict(fut_inp, verbose='auto')
                                tmp_inp.extend(yhat[0].tolist())
                                lst_output.extend(yhat.tolist())
                                i += 1

                        plot_new = np.arange(1, 101)
                        plot_pred = np.arange(101, 101 + forecast_days)
                        
                        st.subheader(f'Forecasted graph of next {forecast_days} days for {model_choice}')  
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=plot_new, y=scaler.inverse_transform(ds_scaled[len(ds_scaled) - 100:]).flatten(), mode='lines', name='Last 100 Days'))
                        fig.add_trace(go.Scatter(x=plot_pred, y=scaler.inverse_transform(lst_output).flatten(), mode='lines', name='Forecast'))
                        fig.update_layout(title=f'Future Values Forecasting - {model_choice}', xaxis_title='Days', yaxis_title='Values')
                        fig.update_layout(plot_bgcolor='white')
                        st.plotly_chart(fig)
                        
                        
                        y_test = y_test.reshape(-1, 1)
                        test_predict = test_predict.reshape(-1, 1)
                        
                        # Calculate RMSE, MSE, MAE, and R^2
                        rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), test_predict))
                        mse = mean_squared_error(scaler.inverse_transform(y_test), test_predict)
                        mae = mean_absolute_error(scaler.inverse_transform(y_test), test_predict)
                        r2 = r2_score(scaler.inverse_transform(y_test), test_predict)
                        
                        # Display evaluation metrics
                        st.write(f"{model_choice} - Root Mean Squared Error (RMSE): {rmse}")
                        st.write(f"{model_choice} - Mean Squared Error (MSE): {mse}")
                        st.write(f"{model_choice} - Mean Absolute Error (MAE): {mae}")
                        st.write("Values closer to 0 for RMSE, MSE, and MAE represent better model performance.")

                        
                        ds_train_inverse = scaler.inverse_transform(ds_scaled.reshape(-1, 1))

                        # Display last 30 values of ds_train_inverse
                        last_few_values = ds_train_inverse[-30:]

                        # Extract the last 30 dates from the original DataFrame
                        last_30_dates = df['date'].iloc[-30:].reset_index(drop=True)

                        # Create a date range for the forecasted values
                        last_date = df['date'].iloc[-1]
                        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(lst_output), freq='D')

                        # Create DataFrame for the last 30 values and their dates
                        last_30_df = pd.DataFrame({'Date': last_30_dates, 'Last 30 Values': last_few_values.flatten()})

                        # Create DataFrame for the forecasted values and their dates
                        forecasted_values = pd.DataFrame({'Date': forecast_dates, 'Forecasted Values': scaler.inverse_transform(lst_output).flatten()})

                        # Concatenate the two DataFrames
                        combined_df = pd.concat([last_30_df, forecasted_values], ignore_index=True)

                        st.write(f"{model_choice} - Last 30 Values and Forecasted Values:")
                        st.write(combined_df)
                        
                    if model_choice == "All Models":
                        for option in model_options[:-1]:  # Exclude "All Models" from options
                            train_and_evaluate_model(option, X_train, y_train, X_test, y_test)
                    else:
                        train_and_evaluate_model(model_choice, X_train, y_train, X_test, y_test)
