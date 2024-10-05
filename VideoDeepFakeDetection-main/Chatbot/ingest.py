
import keras
import nltk
import pickle
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
import random
import datetime
import requests

import sys
sys.stdout.reconfigure(encoding='utf-8')

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

words=[]
classes=[]
documents=[]
ignore=['?','!',',',"'s"]

data_file=open('intents.json').read()
intents=json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#training data
training=[]
output_empty=[0]*len(classes)

for doc in documents:
    bag=[]
    pattern=doc[0]
    pattern=[ lemmatizer.lemmatize(word.lower()) for word in pattern ]
    
    for word in words:
        if word in pattern:
            bag.append(1)
        else:
            bag.append(0)
    output_row=list(output_empty)
    output_row[classes.index(doc[1])]=1
    
    training.append([bag,output_row])
    
random.shuffle(training)
#training=np.array(training)  

X_train = np.array([np.array(bag) for bag, _ in training], dtype=np.float32)
y_train = np.array([np.array(output_row) for _, output_row in training], dtype=np.float32)

# Model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(len(X_train[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

adam = keras.optimizers.Adam(0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
weights = model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=10, verbose=1)
model.save('chatbot_model.h5', weights)
