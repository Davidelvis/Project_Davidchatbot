#importing necessary libraries that will be used
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

#Loading our data

data_file = open('Convo.json').read() #Reading the json file
intents = json.loads(data_file)

#Data preprocessing
#creating lists 
words=[]
classes = []
documents = []
#ignore these words
ignore_words = ['?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenization technique
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #adding documents
        documents.append((w, intent['tag']))

        # append the tags into "classes" list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatization technique
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# this way we can remove duplicates
words = sorted(list(set(words)))
# now sort the "classes" list
classes = sorted(list(set(classes)))

print (len(documents), "documents")
# classes are categories of intents
print (len(classes), "classes", classes)
# print all words after apply the two techniques
print (len(words), "unique lemmatized words", words)

#this will save our 'words' list into new file named 'words.pkl'
pickle.dump(words,open('words.pkl','wb'))

#this will save our 'classes' list into new file named 'classes.pkl'
pickle.dump(classes,open('classes.pkl','wb'))