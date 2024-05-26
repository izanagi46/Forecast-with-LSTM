import streamlit as st

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


# Main Streamlit app
def main():
    st.title('About Time Series Prediction Web App')

    # Introduction
    st.write("""
    This web app was created to provide time series prediction capabilities using various forecasting methods and models.
    """)

    # Creation Process
    st.subheader('Creation Process')
    st.write("""
    This web app was developed by our team of BCA Students of The Heritage Academy, Kolkata. We started by identifying the requirements and objectives of the project, which included:
    
    - Providing a user-friendly interface for time series data exploration and forecasting.
    - Implementing multiple tuned lstm models for accurate predictions.
    - Ensuring scalability and performance to handle large datasets and concurrent users.
    
    We followed an agile development approach, which allowed us to iterate quickly and incorporate feedback from stakeholders throughout the development process.
    """)

    # Frameworks Used
    st.subheader('Frameworks Used')
    st.write("""
    Our web app is built using the following frameworks and technologies:
    
    - **Streamlit:** A Python library for creating interactive web apps with simple Python scripts.
    - **Pandas:** A powerful data manipulation and analysis library in Python.
    - **NumPy:** A fundamental package for scientific computing with Python.
    - **Matplotlib:** A plotting library for creating static, animated, and interactive visualizations in Python.
    - **Plotly:** An open-source graphing library for making interactive, publication-quality graphs online.
    
    These frameworks provided the foundation for building our web app and implementing its various features.
    """)

    # Team Members and Mentor
    st.subheader('Team Members and Mentor')
    st.write("""
    Our team consists of techie individuals with learning enthusiasm in data science, machine learning.
    We would like to acknowledge the following team members for their contributions to this project:
    """)
    
    # Displaying team members' images and names horizontally
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.image("suvojeet_photo.jpg", width=150, caption="Suvojeet Das", use_column_width='always')
        
    with col2:
        st.image("ayushi_photo.jpg", width=150, caption="Ayushi Ranjan Choudhary", use_column_width='always')
        
    with col3:
        st.image("sayantan_photo.jpg", width=150, caption="Sayantan Bhanja", use_column_width='always')
        
    with col4:
        st.image("debargha_photo.jpg", width=150, caption="Debargha Nag", use_column_width='always')

    # Acknowledging the mentor
    st.write("""
    We are also grateful to our mentor, **Prof. Dipankar Das**, for providing guidance and support throughout the development process.
    """)

if __name__ == '__main__':
    main()