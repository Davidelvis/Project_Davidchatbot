#Importing necessary libraries

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random

#importing our data

intents = json.loads(open('intents.json').read())
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

 #Preprocessing
def clean_up_sentence(sentence):
    # tokenization
    sentence_words = nltk.word_tokenize(sentence)
    # lematization
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokeniziation(using the function we created earlier)
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    #return array of nag of words
    return(np.array(bag))

#function for prediction
def predict_class(sentence, model):
    # filtering the prediction based on threshold value
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    #setting threshold value
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort on the basis of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list