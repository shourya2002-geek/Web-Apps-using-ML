import streamlit as st
import pandas as pd


def app():



    uploaded_file = st.sidebar.file_uploader(label="Upload your csv or Excel file",type=['csv','xlsx'],key = "eda")
    if uploaded_file is not None:
        
        df = pd.read_csv(uploaded_file)
        try:
            if st.sidebar.checkbox("Show dataset",False):
                st.write(df)
                
            
        except Exception as e:

            st.write("Please upload the dataset as a csv or an excel file")
        
    else:
        st.write("Please upload the dataset as a csv or an excel file")

        
    