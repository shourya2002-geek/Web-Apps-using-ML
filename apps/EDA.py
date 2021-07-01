
import streamlit as st
import numpy as np
import pandas as pd

def app():
    
    
    uploaded_file = st.sidebar.file_uploader(label="Upload your csv or Excel file",type=['csv','xlsx'],key = "eda")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.subheader("EDA Settings")

        if st.sidebar.checkbox("Numerical Analysis",False):
            st.write(df.describe())   

        if st.sidebar.checkbox("Features",False):
            st.write(df.dtypes)         

        if st.sidebar.checkbox("Type of Variables",False):
            st.write(df.dtypes.astype(str).value_counts()) 
        
        if st.sidebar.checkbox("Preprocessed Dataset",False):
        
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df)
            st.write(scaled_data)

        if st.sidebar.checkbox("NaN values Info",False):
            st.write(df.isnull().sum())  
    