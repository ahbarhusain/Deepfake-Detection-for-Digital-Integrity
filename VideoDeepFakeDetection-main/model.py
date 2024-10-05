from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from datetime import datetime
import json
from time import time as current_time
import importlib
import pickle
import numpy as np
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import random

# Initialize Flask app
app = Flask(__name__)

UPLOAD_FOLDER = 'static/videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize chatbot components
lemmatizer = WordNetLemmatizer()
model = load_model('Chatbot/chatbot_model.h5') 
words = pickle.load(open('Chatbot/words.pkl', 'rb'))
classes = pickle.load(open('Chatbot/classes.pkl', 'rb'))

# Helper function to preprocess user input and generate chatbot response
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# Load intents for responses
intents = json.loads(open('Chatbot/intents.json').read())

# Home route
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/index')
def render_index():
    return render_template('index2.html')

# Route to handle chatbot messages
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")
    intents_list = predict_class(user_message)
    chatbot_response = get_response(intents_list, intents)
    return jsonify({"response": chatbot_response})

# Handle file upload and redirect to the result page
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        timestamp = int(current_time())
        filename = f"uploaded_video_{timestamp}.mp4"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        video_url = url_for('static', filename=f'videos/{filename}')
        module = importlib.import_module("deepfake_detector")
        function = getattr(module, "run")
        result_from_det = function(video_path, video_path)
        print(result_from_det)

        video_info = {
            'name': file.filename,
            'size': f"{os.path.getsize(video_path) / (1024):.2f} KB",
            'user': 'Guest',
            'source': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'per': result_from_det
        }

        video_info_json = json.dumps(video_info)
        return redirect(url_for('result', video_info=video_info_json, video_url=video_url))

@app.route('/result')
def result():
    video_info_json = request.args.get('video_info')
    video_url = request.args.get('video_url')
    video_info = json.loads(video_info_json)
    return render_template('result.html', video_url=video_url, video_info=video_info)

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=False)