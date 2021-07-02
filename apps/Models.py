
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

st.set_option('deprecation.showPyplotGlobalUse', False)

def plot_metrics(model,metrics_list,X_test,Y_test):

    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion matrix")
        plot_confusion_matrix(model,X_test,Y_test)
        st.pyplot()
    
    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model,X_test,Y_test)
        st.pyplot()    

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-recall Curve")
        plot_precision_recall_curve(model,X_test,Y_test)
        st.pyplot() 
      




def app():
    uploaded_file = st.sidebar.file_uploader(label="Upload your csv or Excel file",type=['csv','xlsx'],key = "eda") 
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) 
        prelist = list(df.columns) 
        option = st.sidebar.selectbox(
        'Select the target column for prediction',prelist)
        if st.sidebar.checkbox("Feature counts",False):
            st.write(df[option].value_counts()) 
        def split(df):
            Y = df[option]    
            X = df.drop([option],axis=1)
            X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
            return X_train,X_test,Y_train,Y_test
        
        X_train,X_test,Y_train,Y_test = split(df)
        
        

        option = st.sidebar.selectbox(
        'Select the type of problem',('Classification','Regression')) 
        
        if option == 'Classification':
            classifier = st.sidebar.selectbox("Classifier",("SVM","Logistic Regression","Random Forest","XgBoost"))
            if classifier == 'SVM':
                st.sidebar.subheader("Model Hyperparameters")
                C = st.sidebar.number_input("C (Regularization parameter)",0.01, 10.0, step=0.01)
                kernel = st.sidebar.radio("Kernel", ("rbf","linear"))
                gamma = st.sidebar.radio("Gamma (Kernel coefficient)", ("scale", "auto"))

                metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve', 'Precision-Recall Curve'))
        
                if st.sidebar.button("Classify"):
                    st.subheader("SVM Results")
                    model = SVC(C=C, kernel=kernel ,gamma=gamma)
                    model.fit(X_train,Y_train)
                    accuracy = model.score(X_test,Y_test)
                    Y_pred = model.predict(X_test)
                    st.write("Accuracy: ", accuracy)
                    st.write("Precision: ",precision_score(Y_test,Y_pred,average='micro'))
                    st.write("Recall: ",recall_score(Y_test,Y_pred,average='micro'))
                    plot_metrics(model,metrics,X_test,Y_test)

            if classifier == 'Logistic Regression':
                st.sidebar.subheader("Model Hyperparameters")
                C = st.sidebar.number_input("C (Regularization parameter)",0.01, 10.0, step=0.01)
                max_iter = st.sidebar.slider("Maximum number of iterations",100,500)

                metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve', 'Precision-Recall Curve'))
        
                if st.sidebar.button("Classify"):
                    st.subheader("LR Results")
                    model = LogisticRegression(C=C, max_iter=max_iter)
                    model.fit(X_train,Y_train)
                    accuracy = model.score(X_test,Y_test)
                    Y_pred = model.predict(X_test)
                    st.write("Accuracy: ", accuracy)
                    st.write("Precision: ",precision_score(Y_test,Y_pred,average='micro'))
                    st.write("Recall: ",recall_score(Y_test,Y_pred,average='micro'))
                    plot_metrics(model,metrics,X_test,Y_test)        

            if classifier == 'Random Forest':
                st.sidebar.subheader("Model Hyperparameters")
                n_estimators = st.sidebar.number_input("The number of trees in the classifier", 100, 5000, step=10, key='n_estimators')
                max_depth = st.sidebar.number_input("The maximum depth of a tree", 1, 20, step=1, key='max_depth')
                bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True','False'), key='bootstrap')
                metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve', 'Precision-Recall Curve'))
                
                if st.sidebar.button("Classify"):
                    st.subheader("RF Results")
                    model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap, n_jobs=-1)
                    model.fit(X_train,Y_train)
                    accuracy = model.score(X_test,Y_test)
                    Y_pred = model.predict(X_test)
                    st.write("Accuracy: ", accuracy)
                    st.write("Precision: ",precision_score(Y_test,Y_pred,average='micro'))
                    st.write("Recall: ",recall_score(Y_test,Y_pred,average='micro'))
                    plot_metrics(model,metrics,X_test,Y_test)    

            if classifier == 'XgBoost':
                st.sidebar.subheader("Model Hyperparameters")
                n_estimators = st.sidebar.number_input("The number of trees in the classifier", 100, 5000, step=10, key='n_estimators')
                max_depth = st.sidebar.number_input("The maximum depth of a tree", 1, 20, step=1, key='max_depth')
                eta = st.sidebar.number_input("eta (Learning Rate)",0.2, 0.3, step=0.01)
                metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve', 'Precision-Recall Curve'))
                
                if st.sidebar.button("Classify"):
                    st.subheader("XGB Results")
                    model = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,eta=eta, n_jobs=-1)
                    model.fit(X_train,Y_train)
                    accuracy = model.score(X_test,Y_test)
                    Y_pred = model.predict(X_test)
                    st.write("Accuracy: ", accuracy)
                    st.write("Precision: ",precision_score(Y_test,Y_pred,average='weighted'))
                    st.write("Recall: ",recall_score(Y_test,Y_pred,average='weighted'))
                    plot_metrics(model,metrics,X_test,Y_test)     
        
        
        



    else:
        st.write("Please upload the dataset as a csv or an excel file")        