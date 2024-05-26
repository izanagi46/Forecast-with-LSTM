import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from scipy import stats





st.set_page_config(
    page_title="LSTM Model APP",
    page_icon="😃",
    layout="wide"
)

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


# Add content to your Streamlit app
st.markdown("   #        👋     Welcome To Time Series Forecasting Web App")

# Display the waving GIF
st.image("InShot_20240429_183057206.gif", use_column_width=True)

# Add CSS for animation
st.write("""
    <style>
        @keyframes slide-in {
            0% {
                transform: translateX(-100%);
            }
            100% {
                transform: translateX(0);
            }
        }
        .slide-in-animation {
            animation: slide-in 2.0s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)
# Define the HTML content with CSS for setting background image
background_html = """
    <style>
        body {
            background-image: url("dd_photo.jpg");
            background-size: cover;
            background-position: center;
        }
    </style>
"""

# Render the HTML content
st.markdown(background_html, unsafe_allow_html=True)

# Text with animation
st.write('<div class="slide-in-animation">This web application helps you to gain insights about your given dataset and helps to find the future!!</div>', unsafe_allow_html=True)

st.set_option('deprecation.showPyplotGlobalUse', False)

