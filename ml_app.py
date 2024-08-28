import streamlit as st
import numpy as np

import joblib
import os
import pickle

import re, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

def run_ml_app():
    st.subheader('Text to Predict:')

    text = st.text_area(
        label = "Enter text:",
        value = '',
        height = 500)

    tfidf = TfidfVectorizer(max_features=5000)

    if text:
        X = tfidf.fit_transform([text]).toarray()

        if X.shape[1] < 5000:
            X_padded = np.pad(X, ((0, 0), (0, 5000 - X.shape[1])), mode='constant')
        else:
            X_padded = X[:, :5000]

#    st.write(X_padded)

    st.subheader("Prediction Result:")

    with open('model_SCV_fake_news_prediction.pkl','rb') as file:
        model = pickle.load(file)

    prediction = model.predict(X_padded)

    if prediction == 0:
        st.write('FAKE')
    else:
        st.write('REAL')