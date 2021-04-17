import streamlit as st
import sklearn
import pickle
import pandas as pd
products = pd.read_csv('small_csv_data.csv')
filename = 'Amazon_reviews.sav'
model = pickle.load(open(filename, 'rb'))
st.title('Sentiment Analysis of Amazon Reviews')
ip = st.text_input('Enter your message')
op = model.predict([ip])
if st.button('Predict'):
  st.title(op[0])
st.table(products['Example Reviews'])
