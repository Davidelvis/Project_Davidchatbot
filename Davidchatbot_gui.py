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

    #choosing the response randomly from the predefined reponses for the given identified intent
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

#function to return the response as output in the window
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

#Creat GUI for our chatbot with tkinter library
import tkinter
from tkinter import *

#creating function to send message
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        #we can set the font color, size, family in this way
        ChatLog.config(foreground="white", font=("Verdana", 12))
        #getting the response with the help of function we created earlier
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
       

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 
#creating object
base = Tk()
#set the title
base.title("David Elvis")

#we can change the title icon of tkinter window in this way
from PIL import Image, ImageTk
#this is the image file 
path = "Elvis.jpg"
load = Image.open(path)
render = ImageTk.PhotoImage(load)
base.iconphoto(False, render)
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Creating the Chat window
#We can set the background color and other effects in this way
ChatLog = Text(base, bd=0, bg="#00688B", height="8", width="50", font="Arial")

ChatLog.config(state=DISABLED)

#binding the scroll bar with the chat window
#yview, because we want vertical scroll bar
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#creating send button
#we can set the effects for the same in this manner
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="9", height=3,justify=CENTER,
                    bd=0, bg="#00688B", activebackground="#00688B",fg='#ffffff',
                    command= send )

#box where user will enter the message
EntryBox = Text(base, bd=0, bg="white",width="29", height="3", font="Arial")

#Placing all these components on the screen(adjusting height, width of everything) 
scrollbar.place(x=376,y=6, height=416)
ChatLog.place(x=6,y=6, height=416, width=370)
EntryBox.place(x=6, y=431, height=60, width=265)
SendButton.place(x=273, y=431, height=60)

base.mainloop()