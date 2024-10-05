from flask import Flask, render_template, request
import nltk
from keras.models import load_model
import json
import pickle
import wikipedia

nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('popular')
nltk.download('punkt_tab')
import numpy as np
import random
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

# Load intents, words, classes, and the trained model
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


@app.route("/")
def home():
    return render_template("index.html")


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
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    if not ints:
        return "I'm sorry, I didn't understand that."

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result

    return "I'm sorry, I don't have a response for that."



def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res


@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    bot_text = chatbot_response(user_text)
    return bot_text  # Add this line to return the bot's response


if __name__ == "__main__":
    app.run(debug=True)
