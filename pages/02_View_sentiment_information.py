import streamlit as st 
import pandas as pd
# Load your raw data
raw = pd.read_csv("resources/train.csv")
st.title("This is the sentiment information.")
st.sidebar.success(" View sentiment information")

st.info("General Information")
				# You can read a markdown file from supporting resources folder
st.markdown("Some information here")

st.subheader("Twitter data and its sentiment")
checkbox = st.checkbox('Show raw data')
if checkbox: # data is hidden if box is unchecked
	st.dataframe(raw[['sentiment','message']], height=750)	