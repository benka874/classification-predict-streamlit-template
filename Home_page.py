"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
import warnings
# Data dependencies
import pandas as pd
from PIL import Image
#img = Image.open("streamlit.png")
 
# display image using streamlit
# width is used to set the width of an image
#st.image(img, width=200)
# Vectorizer
#col1, col2, col3 = st.columns([0.5,3,1])
warnings.simplefilter(action='ignore', category=FutureWarning)

#[theme]
#backgroundColor = "#F0F0F0"


st.set_page_config(page_title = "This is a Multipage WebApp") 
st.title("Global Tech Institute.")
st.sidebar.success("Welcome to Global Tech Institution page")
    #st.write(' ')


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	
			# Creating sidebar with selection box -
			# you can create multiple pages this way
	#options = ["Prediction", "Information","Categories","Visuals"]
	#selection = st.sidebar.selectbox("Choose your business service", options)



	# Creates a main title and subheader on your page -
	# these are static across all pages
	
			#st.image("https://static.streamlit.io/examples/dog.jpg")
	img=Image.open("resources/imgs/global1.jpg")
	st.image(img,width=250,use_column_width="always")
	st.title(" Intelligent tweet classifier")
	
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
