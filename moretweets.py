from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import csv
import datetime
import dateutil.parser
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import tensorflow as tf
import tensorflow_hub as hub
from textblob import TextBlob
import sys
import os
import nltk
import pycountry
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import text2emotion as te
import seaborn as sns
import time
import spacy



#VARIABLES TO CHANGE

start_pos = 0 #change to not start at the beginning of your tweet corpus
output_csv = 'output_proc.csv'
match_csv = 'tweet1.csv'# the tweet you want to match
tweets_csv = 'data.csv' #'data.csv' #'CapnMarvel_2.csv' # the corpus of tweets you want to draw from


#Syntactic options
check_syntax = True
min_syntax = .6

#Semantic options
check_semantic = True
min_semantic = .6
check_SBERT = True
check_GUSE = False
min_SBERT = min_GUSE = min_semantic

#Sentiment options
check_sentiment = True
sentiment_type = 'EMOTION' #'STANDARD' OR 'EMOTION' 
min_sentiment = .6

#Target options
check_target = False
min_target = .6

#BEND options
check_bend = False
min_bend = .6


match_array = [check_syntax, check_semantic, check_sentiment, check_target, check_bend]
min_array = [min_syntax, min_semantic, min_sentiment, min_target, min_bend]

# Create output file
csvFile = open(output_csv, "a", newline="", encoding='utf-8')
csvWriter = csv.writer(csvFile)
csvFile.close()

#open ref document
#extract text
df_ref = pd.read_csv(match_csv)
features_ref = ['text']
ref_text = df_ref[features_ref].values

#open draw db
#extract text
df_draw = pd.read_csv(tweets_csv)
features_draw = ['text']
draw_text = df_draw[features_draw].values

# SYNTACTIC+
#word2vec
if check_syntax:
    nlp = spacy.load('en_core_web_md')
    syntax_1 = nlp(ref_text[0][0])

def word2vec_match(queue, syntax1, draw_text, nlp, start_pos):
    import numpy as np
    ranger = len(draw_text)
    for n1 in np.arange(start_pos, ranger):
        syntax2 = nlp(draw_text[n1][0])
        sim_syntax = syntax1.similarity(syntax2)
        queue.put(sim_syntax)

# SEMANTIC

if check_semantic:
    #Setup SBERT
    if check_SBERT:
        multi_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        m_emb1 = multi_model.encode(ref_text[0][0])
    #Setup Google USE
    if check_GUSE:
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
        GUSE_model = hub.load(module_url)

def multilingual_SBERT(queue, memb1, draw_text, min_sem, multi_model, start_pos):
    import numpy as np
    from sentence_transformers import util
    ranger = len(draw_text)
    for n1 in np.arange(start_pos, ranger):
        m_emb2 = multi_model.encode(draw_text[n1][0])
        m_cos_sim = util.cos_sim(memb1, m_emb2)
        queue.put(m_cos_sim.item())

def Google_USE(queue, text1, draw_text, min_sem, GUSE_model, start_pos):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    ranger = len(draw_text)
    for n1 in np.arange(start_pos, ranger):
        text = [text1, draw_text[n1][0]]
        embeddings = GUSE_model(text)
        similarity = cosine_similarity(embeddings)
        queue.put(similarity[0][1].item())

# SENTIMENT
if check_sentiment:
    match sentiment_type:
        case "STANDARD":
            nltk.download('vader_lexicon')
            sid = SentimentIntensityAnalyzer()
            score1b = sid.polarity_scores(ref_text[0][0])
            score_list = [float(score1b['neg']),float(score1b['neu']),float(score1b['pos']),float(score1b['compound'])]
            score1 = np.array(score_list)
        
        case "EMOTION":
            nltk.download('omw-1.4')
            score1b = te.get_emotion(ref_text[0][0])
            score_list = [float(score1b['Happy']),float(score1b['Angry']),float(score1b['Surprise']),float(score1b['Sad']),float(score1b['Fear'])]
            score1 = np.array(score_list)
        case _:
            print("ERROR - YOU DID NOT PROPERLY SPECIFY A METHOD FOR SENTIMENT MATCH. SHOULD BE STANDARD OR EMOTION.")

def sentiment_match(queue, score_1, draw_text, sentiment_type, start_pos):
    #TODO: emotion set takes almost .5s -> fix this
    import text2emotion as te
    import numpy as np
    ranger = len(draw_text)
    for n1 in np.arange(start_pos, ranger):
        match sentiment_type:
            case "STANDARD":
                score_2b = SentimentIntensityAnalyzer().polarity_scores(draw_text[n1][0])
                score2_list = [float(score_2b['neg']),float(score_2b['neu']),float(score_2b['pos']),float(score_2b['compound'])]
                score_2 = np.array(score2_list)
                interim_sim = util.cos_sim(score_1, score_2)
                sentiment_sim = interim_sim[0].item()
            case "EMOTION":
                score2b = te.get_emotion(draw_text[n1][0])
                score2_list = [float(score2b['Happy']),float(score2b['Angry']),float(score2b['Surprise']),float(score2b['Sad']),float(score2b['Fear'])]
                score_2 = np.array(score2_list)
                dist = np.linalg.norm(score_1 - score_2)
                sentiment_sim = 1 - (dist / 2.23606797749979) #normalized by highgest possible distanceand subtracted from 1
            case _:
                print("ERROR - YOU DID NOT PROPERLY SPECIFY A METHOD FOR SENTIMENT MATCH. SHOULD BE STANDARD OR EMOTION.")
        queue.put(sentiment_sim)

if __name__ == '__main__':
    from time import sleep
    from multiprocess import Queue
    from multiprocess import Process
    # We are going to iterate over data that was in a beautiful vector friendly data frame 
    # This is a terrible idea and should never be done
    ranger = len(draw_text)
    queue1 = Queue()
    queue2 = Queue()
    queue3 = Queue()
    queue4 = Queue()

    if check_SBERT: 
        process1 =  Process(target=multilingual_SBERT, args=(queue1, m_emb1, draw_text, min_SBERT, multi_model, start_pos))
        process1.start()
    if check_GUSE:
        process2 = Process(target=Google_USE, args=(queue2, ref_text[0][0], draw_text, min_GUSE, GUSE_model, start_pos))
        process2.start()
    if check_sentiment:
        process3 = Process(target=sentiment_match, args=(queue3, score1, draw_text, sentiment_type, start_pos))
        process3.start()
    if check_syntax:
        process4 = Process(target=word2vec_match, args=(queue4, syntax_1, draw_text, nlp, start_pos))
        process4.start()
        
    start_time = time.time()    
    for n1 in np.arange(start_pos, ranger):
        match_final = True
        interim_time = time.time()
        match_mbert = 0
        match_GUSE = 0
        if check_SBERT: match_mbert = queue1.get()
        if check_GUSE: match_GUSE = queue2.get()
        if check_sentiment: interim2 = queue3.get()
        if check_syntax: interim0 = queue4.get()
        match_array[2] = interim2
        match_array[0] = interim0
        if check_semantic: 
            match_array[1] = max(match_mbert, match_GUSE)
        #iterate over match array
        #match_array = [check_syntax, check_semantic, check_sentiment, check_target, check_bend]
        for n2 in np.arange(len(match_array)):
            if match_array[n2] != False:
                if match_array[n2] >= min_array[n2]:
                    match_final = match_final and True
                else:
                    match_final = False    
        
        if match_final:
            print("FOUND A MATCH WITH TWEET #" + str(n1))
            print(match_array)
            #Open OR create the target CSV file
            csvFile = open(output_csv, "a", newline="", encoding='utf-8')
            csvWriter = csv.writer(csvFile)
            # Append the result to the CSV file
            csvWriter.writerow(df_draw.iloc[n1])
            # When done, close the CSV file
            csvFile.close()
        print("Processed Tweet " + str(n1) + " in " + "%s seconds ---" % (time.time() - interim_time))
    print("--- %s seconds ---" % (time.time() - start_time))
    temp2 = (time.time() - start_time)/ranger
    print("Avg time per tweet: " + str(temp2))