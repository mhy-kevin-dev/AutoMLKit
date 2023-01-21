import streamlit as st
import pandas as pd
import pandas_profiling
from pycaret.regression import setup, compare_models, pull, save_model, load_model, predict_model
from streamlit_pandas_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-03.png")
    st.title("AutoMLKit")
    choice = st.radio("Navigation", ["Upload","Profiling","ML", "Download", "Predict"])
    st.info("This app can help you explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "ML": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    solver = st.selectbox('Choose methods', ("Regression", "Classification"))
    if st.button('Train Model'): 
        st.write("You selected {}".format(solver))
        if solver == "Classification":
            from pycaret.classification import setup, compare_models, pull, save_model, load_model, predict_model

        setup(df, target=chosen_target, silent=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')
        st.subheader("Best: ")
        st.caption(best_model)


if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")


if choice == "Predict": 
    st.title("Upload Your Test Dataset")
    file = st.file_uploader("Upload Your Test Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('test.csv', index=None)
        st.dataframe(df)
        st.subheader("Predicted: ")
        best_model = load_model("best_model")
        predictions = predict_model(best_model, data=df)
        st.dataframe(predictions)
        predictions.to_csv('predict.csv', index=None)
        with open('predict.csv', 'r') as f: 
            st.download_button('Download Predictions', f, file_name="predict.csv")
