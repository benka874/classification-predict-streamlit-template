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
from itertools import count
from symbol import return_stmt
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
from PIL import Image

# Text Processing Libraries
import contractions  # Contractions is used to handle English contractions, converting them into their longer forms.
import emoji  # Emoji allows easy manipulation and analysis of emojis in the text.
from nltk.corpus import stopwords  # Stopwords module provides a list of common words to be removed from the text.
from nltk.stem import WordNetLemmatizer  # WordNetLemmatizer is used for lemmatizing words, bringing them to their root form.
from nltk import download as nltk_download  # For downloading nltk packages, here 'wordnet'.
import regex  # Regex is used for regular expression matching and manipulation.
import string  # Provides constants and classes for string manipulation.
import unicodedata  # Provides access to the Unicode Character Database for processing Unicode characters.
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack  # Used for stacking sparse matrices horizontally.

import nltk
nltk.download('stopwords')
# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

#with open('resources/TFIDF_Vec.pkl', 'rb') as file:
        #tf_vect = pickle.load(file)	
#with open('resources/TFIDF_Vec.pkl', 'rb') as file:
        #tf_vect = pickle.load(file)
		# 		
#new vectorizer
new_count_vec = open("resources/Count_vec.pkl","rb")
count_vec = joblib.load(new_count_vec) # loading your vectorizer from the pkl file
# Load your raw data
raw = pd.read_csv("resources/train.csv")

#load training data
df_train = pd.read_csv('resources/train.csv')

#preprocess function
def preprocess_tweet(tweets):
	# function to determine if there is a retweet within the tweet
	def is_retweet(tweet):
		word_list = tweet.split()
		if "RT" in word_list:
			return 1
		else:
			return 0
	tweets["is_retweet"] = tweets["message"].apply(is_retweet, 1)

	# function to extract retween handles from tweet	
	def get_retweet(tweet):
		word_list = tweet.split()
		if word_list[0] == 'RT':
			handle = word_list[1]
		else:
			handle = ''
		handle = handle.replace(':', "")

		return handle
	tweets['retweet_handle'] = tweets['message'].apply(get_retweet,1)

	# function to count the number of hashtags within the tweet
	def count_hashtag(tweet):
		count = 0

		word_list = tweet.split()
		for word in word_list:
			if word[0] == "#":
				count += 1

		return count
	tweets["hashtag_count"] = tweets["message"].apply(count_hashtag, 1)

	# function to extract the hashtags within the tweet
	def get_hashtag(tweet):
		hashtags = []
		word_list = tweet.split()
		for word in word_list:
			if word[0] == "#":
				hashtags.append(word)

		returnstr = ""

		for tag in hashtags:
			returnstr + " " + tag

		return returnstr
	tweets["hashtags"] = tweets["message"].apply(get_hashtag, 1)

	# function to count the number of mentions within the tweet
	def count_mentions(tweet):
		count = 0
		word_list = tweet.split()
		if "RT" in word_list:
			count += -1 # remove mentions contained in retweet from consideration
		
		for word in word_list:
			if word[0] == '@':
				count += 1
		if count == -1:
			count = 0
		return count
	tweets["mention_count"] = tweets["message"].apply(count_mentions, 1)

	def get_mentions(tweet):
		mentions = []
		word_list = tweet.split()

		if "RT" in word_list:
			word_list.pop(1) # Retweets don't count as mentions, so we remove the retweet handle from consideration

		for word in word_list:
			if word[0] == '@':
				mentions.append(word)

		returnstr = ""

		for handle in mentions:
			returnstr + " " + handle

		return returnstr
	tweets["mentions"] =  tweets["message"].apply(get_mentions, 1)

	# function to count the number of web links within tweet
	def count_links(tweet):
		count = tweet.count("https:")
		return count 
	tweets["link_count"] = tweets["message"].apply(count_links, 1)

	# function to replace URLs within the tweet
	pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
	subs_url = r'url-web'
	tweets['message'] = tweets['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)
	
	# function count number of newlines within tweet
	def enter_count(tweet):
		count = tweet.count('\n')
		return count
	tweets["newline_count"] = tweets["message"].apply(enter_count, 1)

	# function to count number of exclaimation marks within tweet
	def exclamation_count(tweet):
		count = tweet.count('!')
		return count 
	tweets["exclamation_count"] =  tweets["message"].apply(exclamation_count, 1)
	
	# Remove handles from tweet
	def remove_handles(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word[0] == "@":
				wordlist.remove(word)
		returnstr = ''

		for word in wordlist:
			returnstr += word + " "

		return returnstr
	tweets['message'] = tweets['message'].apply(remove_handles)

	# Remove hashtags from tweet
	def remove_hashtags(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word[0] == '#':
				wordlist.remove(word)
		returnstr = ''
		for word in wordlist:
			returnstr += word + " "

		return returnstr
	
	tweets["message"] = tweets["message"].apply(remove_hashtags)

	# Remove RT from tweet
	def remove_rt(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word == 'rt' or word == 'RT':
				wordlist.remove(word)
		returnstr = ''

		for word in wordlist:
			returnstr += word + ' '

		return returnstr
	
	tweets['message'] = tweets['message'].apply(remove_rt)

	# function to translate emojis and emoticons
	def fix_emojis(tweet):
		newtweet = emoji.demojize(tweet)  # Translates 👍 emoji into a form like :thumbs_up: for example
		newtweet = newtweet.replace("_", " ") # Beneficial to split emoji text into multiple words
		newtweet = newtweet.replace(":", " ") # Separate emoji from rest of the words
		returntweet = newtweet.lower() # make sure no capitalisation sneaks in

		return returntweet
	tweets["message"] = tweets['message'].apply(fix_emojis)

	# function to remove punctuation from the tweet
	def remove_punctuation(tweet):
		return ''.join([l for l in tweet if l not in string.punctuation])
	
	tweets['message'] = tweets['message'].apply(remove_punctuation)
	
	#transform tweets into lowercase version of tweets
	def lowercase(tweet):
		return tweet.lower()
	tweets["message"] = tweets["message"].apply(lowercase)

	# remove stop words from the tweet
	def remove_stop_words(tweet):
		words = tweet.split()
		return " " .join([t for t in words if t not in stopwords.words('english')])
	tweets["message"] = tweets['message'].apply(remove_stop_words)

	# function to replace contractions
	def fix_contractions(tweet):
		expanded_words = []
		for word in tweet.split():
			expanded_words.append(contractions.fix(word))

		returnstr = " ".join(expanded_words)
		return returnstr
	tweets["message"] = tweets['message'].apply(fix_contractions)

	# function to replace strange characters in tweet with closest ascii equivalent
	def clean_tweet(tweet):
		normalized_tweet = unicodedata.normalize('NFKD',tweet)

		cleaned_tweet = normalized_tweet.encode('ascii', 'ignore').decode('utf-8')

		return cleaned_tweet.lower()
	tweets["message"] = tweets['message'].apply(clean_tweet)

	#function to remove numbers from tweet
	def remove_numbers(tweet):
		return ''.join(char for char in tweet if not char.isdigit())
	
	tweets["message"] = tweets['message'].apply(remove_numbers)

	# Create a lemmatizer object
	lemmatizer = WordNetLemmatizer()

	# Create function to lemmatize tweet content
	def tweet_lemma(tweet,lemmatizer):
		list_of_lemmas = [lemmatizer.lemmatize(word) for word in tweet.split()]
		return " ".join(list_of_lemmas)
	tweets["message"] = tweets["message"].apply(tweet_lemma, args=(lemmatizer, ))

	# Make dataframe of all word counts in the data
	twt_wordcounts = pd.DataFrame(tweets['message'].str.split(expand=True).stack().value_counts())
	twt_wordcounts.reset_index(inplace=True)
	twt_wordcounts.rename(columns={"index": "word", 0:"count"}, inplace=True)
	
	# Extract unique words from data
	twt_unique_words = twt_wordcounts[twt_wordcounts["count"]==1]

	# make a list of unique words
	unique_wordlist = list(twt_unique_words["word"])

	# Function to remove unique words from data
	def remove_unique_words(tweet):
		words = tweet.split()
		return ' '.join([t for t in words if t not in unique_wordlist])
	tweets['message'] = tweets['message'].apply(remove_unique_words)

	#function to add retweets to message 
	def add_rt_handle(row):
		if row["retweet_handle"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " rt " + row["retweet_handle"]
		return ret
	tweets["message"] = tweets.apply(add_rt_handle, axis=1)

	# Function to add retweets to message
	def add_hashtag(row):
		if row["hashtags"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " " + row["hashtags"]
		return ret
	tweets["message"] = tweets.apply(add_hashtag, axis = 1)

	# Function to add mentions to message
	def add_rt_handle(row):
		if row["mentions"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " " + row["mentions"]
		return ret
	tweets["message"] = tweets.apply(add_rt_handle, axis = 1)

	# drop redundant columns
	tweets = tweets.drop(["retweet_handle", "hashtags", "mentions"], axis=1)

	return tweets

def preprocess_csv(tweets):
	# function to determine if there is a retweet within the tweet
	def is_retweet(tweet):
		word_list = tweet.split()
		if "RT" in word_list:
			return 1
		else:
			return 0
	tweets["is_retweet"] = tweets["message"].apply(is_retweet, 1)

	# function to extract retween handles from tweet	
	def get_retweet(tweet):
		word_list = tweet.split()
		if word_list[0] == 'RT':
			handle = word_list[1]
		else:
			handle = ''
		handle = handle.replace(':', "")

		return handle
	tweets['retweet_handle'] = tweets['message'].apply(get_retweet,1)

	# function to count the number of hashtags within the tweet
	def count_hashtag(tweet):
		count = 0
		word_list = tweet.split()
		for word in word_list:
			if word[0] == "#":
				count += 1

		return count
	tweets["hashtag_count"] = tweets["message"].apply(count_hashtag, 1)

	# function to extract the hashtags within the tweet
	def get_hashtag(tweet):
		hashtags = []
		word_list = tweet.split()
		for word in word_list:
			if word[0] == "#":
				hashtags.append(word)

		returnstr = ""

		for tag in hashtags:
			returnstr + " " + tag

		return returnstr
	tweets["hashtags"] = tweets["message"].apply(get_hashtag, 1)

	# function to count the number of mentions within the tweet
	def count_mentions(tweet):
		count = 0
		word_list = tweet.split()
		if "RT" in word_list:
			count += -1 # remove mentions contained in retweet from consideration
		
		for word in word_list:
			if word[0] == '@':
				count += 1
		if count == -1:
			count = 0
		return count
	tweets["mention_count"] = tweets["message"].apply(count_mentions, 1)

	def get_mentions(tweet):
		mentions = []
		word_list = tweet.split()

		if "RT" in word_list:
			word_list.pop(1) # Retweets don't count as mentions, so we remove the retweet handle from consideration

		for word in word_list:
			if word[0] == '@':
				mentions.append(word)

		returnstr = ""

		for handle in mentions:
			returnstr + " " + handle

		return returnstr
	tweets["mentions"] =  tweets["message"].apply(get_mentions, 1)

	# function to count the number of web links within tweet
	def count_links(tweet):
		count = tweet.count("https:")
		return count 
	tweets["link_count"] = tweets["message"].apply(count_links, 1)

	# function to replace URLs within the tweet
	pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
	subs_url = r'url-web'
	tweets['message'] = tweets['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)
	
	# function count number of newlines within tweet
	def enter_count(tweet):
		count = tweet.count('\n')
		return count
	tweets["newline_count"] = tweets["message"].apply(enter_count, 1)

	# function to count number of exclaimation marks within tweet
	def exclamation_count(tweet):
		count = tweet.count('!')
		return count 
	tweets["exclamation_count"] =  tweets["message"].apply(exclamation_count, 1)
	
	# Remove handles from tweet
	def remove_handles(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word[0] == "@":
				wordlist.remove(word)
		returnstr = ''

		for word in wordlist:
			returnstr += word + " "

		return returnstr
	tweets['message'] = tweets['message'].apply(remove_handles)

	# Remove hashtags from tweet
	def remove_hashtags(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word[0] == '#':
				wordlist.remove(word)
		returnstr = ''
		for word in wordlist:
			returnstr += word + " "

		return returnstr
	
	tweets["message"] = tweets["message"].apply(remove_hashtags)

	# Remove RT from tweet
	def remove_rt(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word == 'rt' or word == 'RT':
				wordlist.remove(word)
		returnstr = ''

		for word in wordlist:
			returnstr += word + ' '

		return returnstr
	
	tweets['message'] = tweets['message'].apply(remove_rt)

	# function to translate emojis and emoticons
	def fix_emojis(tweet):
		newtweet = emoji.demojize(tweet)  # Translates 👍 emoji into a form like :thumbs_up: for example
		newtweet = newtweet.replace("_", " ") # Beneficial to split emoji text into multiple words
		newtweet = newtweet.replace(":", " ") # Separate emoji from rest of the words
		returntweet = newtweet.lower() # make sure no capitalisation sneaks in

		return returntweet
	tweets["message"] = tweets['message'].apply(fix_emojis)

	# function to remove punctuation from the tweet
	def remove_punctuation(tweet):
		return ''.join([l for l in tweet if l not in string.punctuation])
	
	tweets['message'] = tweets['message'].apply(remove_punctuation)
	
	#transform tweets into lowercase version of tweets
	def lowercase(tweet):
		return tweet.lower()
	tweets["message"] = tweets["message"].apply(lowercase)

	# remove stop words from the tweet
	def remove_stop_words(tweet):
		words = tweet.split()
		return " " .join([t for t in words if t not in stopwords.words('english')])
	tweets["message"] = tweets['message'].apply(remove_stop_words)

	# function to replace contractions
	def fix_contractions(tweet):
		expanded_words = []
		for word in tweet.split():
			expanded_words.append(contractions.fix(word))

		returnstr = " ".join(expanded_words)
		return returnstr
	tweets["message"] = tweets['message'].apply(fix_contractions)

	# function to replace strange characters in tweet with closest ascii equivalent
	def clean_tweet(tweet):
		normalized_tweet = unicodedata.normalize('NFKD',tweet)

		cleaned_tweet = normalized_tweet.encode('ascii', 'ignore').decode('utf-8')

		return cleaned_tweet.lower()
	tweets["message"] = tweets['message'].apply(clean_tweet)

	#function to remove numbers from tweet
	def remove_numbers(tweet):
		return ''.join(char for char in tweet if not char.isdigit())
	
	tweets["message"] = tweets['message'].apply(remove_numbers)

	# Create a lemmatizer object
	lemmatizer = WordNetLemmatizer()

	# Create function to lemmatize tweet content
	def tweet_lemma(tweet,lemmatizer):
		list_of_lemmas = [lemmatizer.lemmatize(word) for word in tweet.split()]
		return " ".join(list_of_lemmas)
	tweets["message"] = tweets["message"].apply(tweet_lemma, args=(lemmatizer, ))

	# Make dataframe of all word counts in the data
	twt_wordcounts = pd.DataFrame(tweets['message'].str.split(expand=True).stack().value_counts())
	twt_wordcounts.reset_index(inplace=True)
	twt_wordcounts.rename(columns={"index": "word", 0:"count"}, inplace=True)
	
	# Extract unique words from data
	twt_unique_words = twt_wordcounts[twt_wordcounts["count"]==1]

	# make a list of unique words
	unique_wordlist = list(twt_unique_words["word"])

	# Function to remove unique words from data
	def remove_unique_words(tweet):
		words = tweet.split()
		return ' '.join([t for t in words if t not in unique_wordlist])
	tweets['message'] = tweets['message'].apply(remove_unique_words)

	#function to add retweets to message 
	def add_rt_handle(row):
		if row["retweet_handle"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " rt " + row["retweet_handle"]
		return ret
	tweets["message"] = tweets.apply(add_rt_handle, axis=1)

	# Function to add retweets to message
	def add_hashtag(row):
		if row["hashtags"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " " + row["hashtags"]
		return ret
	tweets["message"] = tweets.apply(add_hashtag, axis = 1)

	# Function to add mentions to message
	def add_rt_handle(row):
		if row["mentions"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " " + row["mentions"]
		return ret
	tweets["message"] = tweets.apply(add_rt_handle, axis = 1)

	# drop redundant columns
	tweets = tweets.drop(["retweet_handle", "hashtags", "mentions"], axis=1)

	return tweets
# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages

	image = Image.open('resources/imgs/logo.png')

	col1, col2 = st.columns([3, 3])
	with col1:
		st.image(image, use_column_width=True)
	with col2:
		st.title("Twitter Sentiment Classifier App")
	#add more text

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	st.sidebar.title('App Navigation')

	options = ["Sentiment Prediction", "About Us", "Model Explanations", "Data Overview"]
	selection = st.sidebar.radio("Choose Option", options)

	#build out the "home" company page
	if selection == "About Us":
		st.info('We are African Global Institute of Technology')
		st.markdown("Established in 2012, the African Global Institute of Technology (AGIT) is a visionary institution dedicated to advancing technology solutions with a global perspective. In the rapidly evolving digital landscape, AGIT recognized the transformative power of technology as a driving force for positive change worldwide. From our inception, AGIT has been committed to fostering innovation, education, and entrepreneurship on a global scale. Rooted in a belief that technology can address complex challenges and contribute to sustainable development, AGIT has evolved into a dynamic hub of excellence, inclusivity, and sustainability. ")
		st.write('To access the codebase for this application, please visit the following GitHub repository:https://github.com/benka874/Classification-Sprint-2307ACDS-Team-EG2.git')

		st.subheader('Meet the team')

		# Professor in Artificial Intelligence and Machine Learning: Benson Kamau
		col1, col2 = st.columns([1, 6])
		with col1:
			image_k = Image.open('resources/imgs/Benson.png')
			st.image(image_k, use_column_width=True,caption = 'Professor in Artificial Intelligence and Machine Learning: Benson Kamau')

		# Professor in Sustainable Technology and Environmental Sciences: Christelle Coetzee
		col1, col2 = st.columns([1, 6])
		with col1:
			image_m = Image.open('resources/imgs/Christelle.png')
			st.image(image_m, use_column_width=True, caption = 'Professor in Sustainable Technology and Environmental Sciences: Christelle Coetzee')

		# Professor in  Global Innovation and Entrepreneurship: Tonia Omonayin
		col1, col2 = st.columns([1, 6])
		with col1:
			image_h = Image.open('resources/imgs/Tonia.png')
			st.image(image_h, use_column_width=True, caption = 'Professor in  Global Innovation and Entrepreneurship: Tonia Omonayin')
		
		# PhD in Data Science and Analytics: Mpho Sesinyi
		col1, col2 = st.columns([1, 6])
		with col1:
			image_t = Image.open('resources/imgs/Mpho.png')
			st.image(image_t, use_column_width=True, caption = 'PhD in Data Science and Analytics: Mpho Sesinyi')

		# PhD in Cybersecurity and Digital Privacy: Joseph Mhlomi
		col1, col2 = st.columns([1, 6])
		with col1:
			image_kg = Image.open('resources/imgs/Joseph.png')
			st.image(image_kg, use_column_width=True, caption = 'PhD in Cybersecurity and Digital Privacy: Joseph Mhlomi')

		# PhD in Human-Computer Interaction: Nozie Bhila
		col1, col2 = st.columns([1, 6])
		with col1:
			image_i = Image.open('resources/imgs/Nozie.png')
			st.image(image_i, use_column_width=True, caption = 'PhD in Human-Computer Interaction: Nozie Bhila')


	# Building out the "Model Explaination" page
	if selection == "Model Explanations":
		options = ['Logistic Regression','Linear Support Vector Classifier', 'Guassian Naives Bayes Classifier']
		selection = st.selectbox('Which model would you like to learn more about?',options)

		if selection == "Logistic Regression":
			#st.info('Explain the inner workings of Logistic Regression model')
			st.markdown("Let's imagine you're in a fruit market, and you want to predict the type of fruit (apple, banana, or orange) based on its color. Multiclass Logistic Regression is like having a smart guide who quickly learns from the colors of different fruits to make predictions about new ones.")
			st.markdown("In Multiclass Logistic Regression, we're dealing with multiple classes, but the basic idea remains similar to binary logistic regression. The goal is to connect the color of a fruit with the likelihood of it being an apple, banana, or orange. Instead of a traffic light, now we have a set of scales—one for each type of fruit.")
			st.markdown('The magic happens through the same logistic function, but this time, we have multiple settings for each type of fruit. Each scale considers the probability of the fruit belonging to its specific category.')
			st.markdown('Imagine you have three scales for apples, bananas, and oranges. The color of a new fruit is plugged into these scales, and the model calculates the probability for each type. If the apple scale tips higher, it suggests the fruit is likely an apple.')
			st.markdown("The learning process involves adjusting the internal settings (weights and biases) of these scales based on past fruits' colors. It's like tuning each scale to be more accurate in predicting the type of fruit.")
			st.markdown("Multiclass Logistic Regression is like having a knowledgeable guide at the fruit market. It quickly learns which colors are typical for apples, bananas, and oranges, tunes its scales accordingly, and uses this knowledge to predict the type of a new fruit. It's a handy tool for classifying data into multiple categories based on given features.")
			
			
		if selection == "Linear Support Vector Classifier":
			#st.info('Explain the inner workings of Support Vector Machines model')
			st.markdown('Imagine you have a set of points on a graph, and you want to draw a straight line (or a plane in more complex cases) to separate those points into different groups, like putting apples on one side and oranges on the other. This line is what we call a hyperplane.')
			st.markdown('The goal of a Linear SVM is to find the best possible line that creates the biggest gap (margin) between the two groups of points. This gap is essential because it helps our model make predictions more confidently and accurately when faced with new, unseen points.')
			st.markdown('In this process, some points in the dataset play a special role; they are called support vectors. They are the ones sitting right at the edge of the gap, helping to define the line and the margin. The Linear SVM ensures these support vectors are strategically chosen to create the most effective separation.')
			st.markdown('The math behind it involves figuring out the best weights and biases to draw that separation line. Think of it as finding the perfect recipe for drawing that line so that it accurately classifies points into their respective groups.')
			st.markdown("In the end, the Linear SVM provides us with a decision function. When we have a new point, we plug it into this function, and if the result is positive, we say it belongs to one group; if it's negative, it belongs to the other.")
			st.markdown("In a nutshell, Linear SVM is like finding the best line to separate different groups of points in a way that makes future predictions as reliable as possible. It's a fundamental tool in machine learning, helping us make sense of data and make predictions in a variety of applications.")
			

		if selection == "Guassian Naives Bayes Classifier":
			#st.info('Explain the inner workings of Naives Bayes model')
			st.markdown("Imagine you're trying to figure out if an email is spam or not based on the words it contains. Gaussian Naive Bayes is like having a clever assistant who quickly learns from past emails to decide if a new one is likely to be spam or not.")
			st.markdown("In Gaussian Naive Bayes, we're dealing with probabilities, but this time, we're considering the distribution of words in spam and non-spam emails. It assumes that the presence of each word is independent of the others—hence, the term 'naive.'")
			st.markdown("Let's break it down: Suppose we have two types of emails, spam and non-spam. Gaussian Naive Bayes looks at the distribution of word frequencies in both types. For each word, it calculates the average frequency and how spread out the frequencies are (the standard deviation).")
			st.markdown("Now, when a new email arrives, our assistant calculates the probability of it being spam or non-spam based on the distribution it learned. It's like saying, 'Given that this email has the word 'discount' and 'urgent,' what's the chance it's spam?'")
			st.markdown("The math involves a bit of statistics, but the idea is intuitive. It's like having a mental scale: if the word 'discount' usually appears more often in spam emails, seeing it in a new email tips the scale toward it being spam.")
			st.markdown("So, Gaussian Naive Bayes is a bit like a detective who quickly assesses the words in an email, compares them to what it learned from past spam and non-spam emails, and makes an educated guess about whether the new email is spam or not. It's a handy tool for text classification and spam filtering, making decisions based on the likelihood of certain features appearing together.")
			
			#st.markdown('Explain the inner workings of this model')

		
	if selection == "Data Overview":
		options =  ['Dataset','Distribution of data per sentiment class','Proportion of retweets','Popular retweet handles per sentiment group in a word cloud', 'Popular hashtags in per sentiments group','Popular mentions per sentiment group']
		selection = st.selectbox('What would like to explore?', options)

		if selection == 'Dataset':
			st.subheader('Overview of dataset:')
			st.write(df_train.head(10))
			st.info("The dataset consists of three columns namely: Sentiment, Tweet and Tweet ID")
			st.markdown("The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43,943 tweets were collected. Each tweet is labelled as one of 4 classes.")
		if selection == 'Distribution of data per sentiment class':
			st.subheader('Distribution of data per sentiment class')
			st.image('resources/imgs/distribution_of_data_in_each_class.png')
			st.markdown('From the figures above, we see that the dataset we are working with is very unbalanced. More than half of our dataset is people having pro-climate change sentiments, while only  8% of our data represents people with anti-climate change opinions. This might lead our models to become far better at identifying pro-climate change sentiment than anti-climate change sentiment, and we might need to consider balancing the data by resampling it.')

		if selection == 'Proportion of retweets':
			st.subheader('Proportion of retweets')
			st.image('resources/imgs/proportion_of_retweets_hashtags_and_original_mentions.png')
			st.markdown('We see that most of our data is not original tweets, but retweets! This indicates that extracting more information from the retweets could prove integral to optimizing our model\'s predictive capabilities.')

		if selection == 'Popular retweet handles per sentiment group in a word cloud':
			st.subheader('Popular retweet handles per sentiment group in a word cloud')
			st.image('resources/imgs/wordcloud_of_popular_retweet_handles_per_sentiment_group.png')
			st.markdown('From the above, we see a clear difference between every sentiment with regards to who they are retweeting. This is great news, since it will provide an excellent feature within our model. Little overlap between categories is visible, which points to the fact that this feature could be a very strong predictor.')
			st.markdown('We see that people with anti-climate change sentiments retweets from users like @realDonaldTrump and @SteveSGoddard the most. Overall retweets associated with anti-climate science opinions are frequently sourced from prominent Republican figures such as Donald Trump, along with individuals who identify as climate change deniers, like Steve Goddard.')
			st.markdown('In contrast to this, people with pro-climate change views often retweet Democratic political figures such as @SenSanders and @KamalaHarris. Along with this, we see a trend to retweet comedians like @SethMacFarlane. The most retweeted individual for this category, is @StephenSchlegel.')
			st.markdown('Retweets in the factual news category mostly contains handles of media news organizations, like @thehill, @CNN, @wasgingtonpost etc...')
			st.markdown('People with neutral sentiments regarding climate change seems to not retweet overtly political figures. Instead, they retweet handles unknown to the writer like @CivilJustUs and @ULTRAVIOLENCE which no longer currently exist on twitter. The comedian @jay_zimmer is also a common retweeted incividual within this category.')

		if selection == 'Popular hashtags in per sentiments group':
			st.subheader('Popular hashtags in per sentiments group')
			st.image('resources/imgs/popular_hashtags_per_sentiment_group_wordcloud.png')
			st.markdown('From the visual above, we notice a few things:')
			st.markdown('We see that a lot of hashtags are common in every sentiment category. Hashtags like #climatechange, #cllimate and #Trump is abundant regardless of which category. However, #Trump is more often used in the anti-cliamte sentiment and #climate and #climatechange is more often used in the factual news sentiment. This can help our model place the tweet into the correct category.')
			st.markdown('Finally there is some hashtags that are more prominent within certain sentiment groups. Take #MAGA and #fakenews in the anti-climate change category, or #LeoDiCaprio and #ParisAgreement in the pro-climate change category. This indicates that some useful information can be extracted from this feature, and should remain within the model.')

		if selection == 'Popular mentions per sentiment group':
			st.subheader('Popular mentions per sentiment group')
			st.image('resources/imgs/popular_hashtags_per_sentiment_group_wordcloud.png')
			st.markdown('As was the case when we considered hashtags, we see that some handles get mentioned regardless of sentiment class. An example of this is @realDonaldTrump, which is prominent in every sentiment category, however it it appears more often in the anti-climate change sentiment. This can help the model distingush from the sentiments.')
			st.markdown('Furthermore, there is some mentions that are more prominent in certain classes than others. Take @LeoDiCaprio for example, which features heavily in both pro-climate change as well as neutral towards climate change sentiment, but is not represented in the other two categories. This indicates that this feature could be beneficial for categorizing our data, and should remain within the dataset')

		
	# Building out the predication page
	if selection == 'Sentiment Prediction':
		st.write('Predict the sentiment of each Twitter post through diverse models, categorizing each tweet into one of four classes: anti-climate change, neutral, pro-climate change, and lastly - factual news.')

		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Tweet Here:","Type Here")

		options = ["Logistic Regression Classifier", "Linear Support Vector Classifier", "Gaussian Naives Bayes Classifier"] 
		selection = st.selectbox("Choose Your Model", options)

		if st.button("Classify Tweet"):
			#process single tweet using our preprocess_tweet() function

			# create dataframe for tweet
			text = [tweet_text]
			df_tweet = pd.DataFrame(text, columns=['message'])

			processed_tweet = preprocess_tweet(df_tweet)
				
			# Create a dictionary for tweet prediction outputs
			dictionary_tweets = {'[-1]': "This tweet is anti-climate change.",
                     			'[0]': "This tweet is neutral and neither supporting nor refuting the belief of climate change.",
                     			'[1]': "This tweet pro-climate change.",
                     			'[2]': "This tweet refers to factual news about climate change"}

			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = None
			X_pred = None
			# if selection == "Multinomial Naive Bayes Classifier":
			# 	predictor = joblib.load(open(os.path.join("resources/MultinomialNaiveBeyes.pkl"),"rb"))
					#mnb = pickle.load(open('/resources/MultinomialNaiveBeyes.pkl','rb'))
					#predictor = mnb	
			if selection == "Logistic Regression Classifier":
				#lr = pickle.load(open('\resources\LogisticRegression.pkl','rb'))
					predictor = joblib.load(open(os.path.join("resources/LogisticRegression.pkl"),"rb"))
			elif selection == "Linear Support Vector Classifier":
				#lsvc = pickle.load(open('/resources/LinearSVC.pkl','rb'))
				predictor = joblib.load(open(os.path.join("resources/LinearSVC.pkl"),"rb"))
				#predictor = lsvc
			# elif selection == "XGBoost Classifier":
			# 	predictor = joblib.load(open(os.path.join("resources/XGBoost.pkl"),"rb"))
			elif selection == "Gaussian Naives Bayes Classifier":
				predictor = joblib.load(open(os.path.join("resources/GaussianNaiveBeyes.pkl"),"rb"))
			

				
			# Transforming user input with vectorizer
			X_pred = processed_tweet['message']
			vect_text = count_vec.transform(X_pred)
				
			sparse_vec_msg_df = pd.DataFrame.sparse.from_spmatrix(vect_text, columns = count_vec.get_feature_names_out())
			df_vectorized_combined = pd.concat([processed_tweet.reset_index(drop=True), sparse_vec_msg_df.reset_index(drop=True)], axis=1)
			df_vectorized_combined = df_vectorized_combined.drop("message", axis='columns')

			prediction = predictor.predict(df_vectorized_combined)
			prediction_str = np.array_str(prediction)

			prediction_str = prediction_str.replace(".","")
				
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(dictionary_tweets[prediction_str]))
			# tweets_csv = st.file_uploader('Upload a CSV file here', type='csv', accept_multiple_files=False, key=None, help='Only CSV files are accepted', on_change=None, args=None, kwargs=None)
			# df_uploaded = None
			# X_pred = None
			# if tweets_csv is not None:
			# 	df_uploaded = pd.read_csv(tweets_csv)
			# 	processed_df = preprocess_csv(df_uploaded)
			# 	X_pred = processed_df['message']
			
			# options = [" Multinomial Naive Bayes Classifier","Logistic Regression Classifier", "Linear Support Vector Classifier"]
			# selection = st.selectbox("Choose Your Model", options)

			# if st.button("Classify CSV"):
			# 	# Transforming user input with vectorizer
			# 	#vect_text = tweet_cv.transform([tweet_text]).toarray()
			# 	# Load your .pkl file with the model of your choice + make predictions
			# 	# Try loading in multiple models to give the user a choice
			# 	predictor = None
			# 	if selection == "Multinomial Naive Bayes Classifier":
			# 		mnb = pickle.load(open('/resources/MultinomialNaiveBeyes.pkl','rb'))
			# 		predictor = mnb	
			# 	elif selection == "Logistic Regression Classifier":
			# 		#lr = pickle.load(open('/resources/LogisticRegression.pkl','rb'))
			# 		predictor = joblib.load(open(os.path.join("resources/LogisticRegression.pkl"),"rb"))
			# 	elif selection == "Linear Support Vector Classifier":
			# 		lsvc = pickle.load(open('resources/LinearSVC.pkl','rb'))
			# 		predictor = lsvc

			# 	#predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			# 	#processed_df = preprocess_csv(df_uploaded)
			# 	vect_text = count_vec.transform(X_pred)
			# 	sparse_vec_msg_df = pd.DataFrame.sparse.from_spmatrix(vect_text, columns = count_vec.get_feature_names_out())
			# 	df_vectorized_combined = pd.concat([processed_df.reset_index(drop=True), sparse_vec_msg_df.reset_index(drop=True)], axis=1)

			# 	df_vectorized_combined = df_vectorized_combined.drop(["tweetid","message"], axis='columns')


			# 	prediction = predictor.predict(df_vectorized_combined)
			# 	df_download = df_uploaded.copy()
			# 	df_download['sentiment'] = prediction

			# 	# When model has successfully run, will print prediction
			# 	# You can use a dictionary or similar structure to make this output
			# 	# more human interpretable.

			# 	#st.success("Text Categorized as: {}".format(prediction))
			# 	st.success("Tweets succesfully classified")
			# 	st.dataframe(data=df_download, width=None, height=None)
			# 	st.download_button(label='Download CSV with sentiment predictions', data=df_download.to_csv(),file_name='sentiment_predictions.csv',mime='text/csv')

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
