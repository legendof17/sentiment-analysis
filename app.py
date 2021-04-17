import streamlit as st
import sklearn
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
products = pd.read_csv('small_csv_data.csv')
filename = 'Amazon_reviews'
model = Pipeline([('tfidf',TfidfVectorizer()),('model',SVC())])
model = joblib.load(filename)
st.title('Sentiment Analysis of Amazon Reviews')
ip = st.text_input('Enter your message')
op = model.predict([ip])
if st.button('Predict'):
  st.title(op[0])
st.table(products['Example Reviews'])
