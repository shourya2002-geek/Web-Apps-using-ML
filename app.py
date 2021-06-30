import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from sklearn.preprocessing import StandardScaler


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

uploaded_file = st.sidebar.file_uploader(label="Upload your csv or Excel file",type=['csv','xlsx'])






if uploaded_file is not None:

    global df
    global lis
    
    try:
        df = pd.read_csv(uploaded_file)
        
    except Exception as e:
        
        df = pd.read_excel(uploaded_file)

    
    try:
        if st.sidebar.checkbox("Show dataset",False):
            st.write(df) 
            lis = list(df.select_dtypes(['float','int']).columns) 
            
    except Exception as e:
        
        st.write("Please upload the dataset as a csv or an excel file")



    st.sidebar.subheader("Plot Settings")
    

    if st.sidebar.checkbox("Avail Plots",False):
        chart_select = st.sidebar.selectbox(
        label="Select the chart type",
        options=['Scatterplots', 'Lineplots' , 'Histogram' , 'Boxplot']
        )
        if chart_select == 'Scatterplots':
        
            try:
                x_values = st.sidebar.selectbox('X axis', options=lis)
                y_values = st.sidebar.selectbox('Y axis', options=lis)
                plot = px.scatter(data_frame=df, x=x_values, y=y_values)
                st.plotly_chart(plot)

            except Exception as e:
                print(e)    


        if chart_select == 'Lineplots':
            
            try:
                x_values = st.sidebar.selectbox('X axis', options=lis)
                y_values = st.sidebar.selectbox('Y axis', options=lis)
                plot = px.line(data_frame=df, x=x_values, y=y_values)
                st.plotly_chart(plot)

            except Exception as e:
                print(e)  

        if chart_select == 'Histogram':
            
            try:
                x_values = st.sidebar.selectbox('X axis', options=lis)
                y_values = st.sidebar.selectbox('Y axis', options=lis)
                plot = px.histogram(data_frame=df, x=x_values, y=y_values)
                st.plotly_chart(plot)

            except Exception as e:
                print(e)          

        if chart_select == 'Boxplot':
            
            try:
                x_values = st.sidebar.selectbox('X axis', options=lis)
                y_values = st.sidebar.selectbox('Y axis', options=lis)
                plot = px.box(data_frame=df, x=x_values, y=y_values)
                st.plotly_chart(plot)

            except Exception as e:
                print(e)   
            


    st.sidebar.subheader("EDA Settings")

    if st.sidebar.checkbox("Type of Variables",False):
        st.write(df.dtypes.astype(str).value_counts()) 
    
    if st.sidebar.checkbox("Show the preprocessed Dataset",False):
       
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        st.write(scaled_data)

    if st.sidebar.checkbox("NaN values info",False):
        
        st.write(df.isnull().sum())           

            



    