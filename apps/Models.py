import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split





def app():
    def split(df):
        Y = df[option]    
        X = df.drop([option],axis=1)
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
        return X_train,X_test,Y_train,Y_test
    X_train,X_test,Y_train,Y_test = split(df)


    option = st.sidebar.selectbox(
    'Select the type of problem',('Classification','Regression')) 
    
    if option == 'Classification':
        classifier = st.sidebar.selectbox("Classifier",("SVM","Logistic Regression"))
        
    if option == 'Regression':
         classifier = st.sidebar.selectbox("Regressor",("XgBoost","Linear Regression"))    