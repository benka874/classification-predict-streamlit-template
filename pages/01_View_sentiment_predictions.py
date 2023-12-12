import streamlit as st 
import joblib,os
import warnings
import pandas as pd  
st.title("Type your  tweet below")
st.sidebar.success(" View your tweet predictions")

news_vectorizer = open("resources/tf_idf_vector.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")


st.subheader("Climate change tweet classification according to sentiment")

			

		#st.write(raw[['sentiment', 'message']]) # will write the df to the page
						
			# Building out the predication page
	#if selection == "Prediction":
st.info("Prediction with ML Models")
				# Creating a text box for user input
tweet_text = st.text_area("Enter Text")
	
if st.button("Classify"):
							# Transforming user input with vectorizer
          vect_text = tweet_cv.transform([tweet_text]).toarray()
							# Load your .pkl file with the model of your choice + make predictions
							# Try loading in multiple models to give the user a choice
          predictor = joblib.load(open(os.path.join("resources/rf_model.pkl"),"rb"))
          prediction = predictor.predict(vect_text)

							# When model has successfully run, will print prediction
							# You can use a dictionary or similar structure to make this output
							# more human interpretable.
          st.success("Text Categorized as: {}".format(prediction))
