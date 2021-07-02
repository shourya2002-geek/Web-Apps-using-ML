import streamlit as st
import pandas as pd
import numpy as np
from multiapp import MultiApp
from apps import Plot, EDA, Models, Main
import pickle
import os

hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)






st.title("Data Analysis App")

st.sidebar.header("Analytical Settings")


app = MultiApp()   

app.add_app("Show Raw Dataset",Main.app)
app.add_app("Exploratory Data Analysis", EDA.app)
app.add_app("Plots", Plot.app)
app.add_app("Model Training", Models.app)


app.run() 
                        
  

    


        



    

        
    
       
            



    