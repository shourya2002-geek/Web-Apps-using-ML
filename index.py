import streamlit as st
import pandas as pd
import numpy as np
from multiapp import MultiApp
from apps import Plot, EDA


hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)




st.title("Data Visualization App")

st.sidebar.header("Visualization Settings")

uploaded_file = st.sidebar.file_uploader(label="Upload your csv or Excel file",type=['csv','xlsx'],key = "index")

global df

if uploaded_file is not None:
    
    
    try:
        df = pd.read_csv(uploaded_file)
        
    except Exception as e:
        
        df = pd.read_excel(uploaded_file)

    try:
        if st.sidebar.checkbox("Show dataset",False):
            st.write(df) 
            
            
    except Exception as e:
        
        st.write("Please upload the dataset as a csv or an excel file")

    app = MultiApp()   
    app.add_app("Plots", Plot.app)
    app.add_app("Exploratory Data Analysis", EDA.app)
    
    app.run() 
                        
  

    


        



    

        
    
       
            



    