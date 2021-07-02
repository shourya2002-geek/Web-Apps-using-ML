
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pickle






def app():
    
    
    uploaded_file = st.sidebar.file_uploader(label="Upload your csv or Excel file",type=['csv','xlsx'],key = "eda") 
    if uploaded_file is not None:   
        
        df = pd.read_csv(uploaded_file)  
        lis = list(df.select_dtypes(['float','int']).columns) 
         
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
    else:
        st.write("Please upload the dataset as a csv or an excel file")                 
        
