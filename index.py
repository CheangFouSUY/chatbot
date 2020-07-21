import random
import json
from keras.models import load_model
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
import anime
import re

# from tkinter import *
# import tkinter

import nltk
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
words_with_tags = pickle.load(open('words_with_tags.pkl', 'rb'))

matched_patterns = []


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    matched_patterns.clear()
    matched_patterns.extend(bag)

    return(np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.6
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    if not results:
        return [{'intent': 'noanswer', 'probability': '0.9'}]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        matched_tag = classes[r[0]]
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    matched_patterns_with_tags = []

    for idx, matched in enumerate(matched_patterns):
        if matched and words_with_tags[idx]['tag'] == matched_tag:
            # matched_patterns_with_tags.append(words_with_tags[idx]['pattern'])
            print(words_with_tags[idx]['pattern'])
            matched_patterns_with_tags.append(str(idx))

    # print(matched_patterns_with_tags)

    return matched_patterns_with_tags


# def getResponse(ints, intents_json):
#     tag = ints[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if(i['tag'] == tag):
#             result = random.choice(i['responses'])
#             break

#     return result

def getResponse(patterns, name=""):
    if not patterns:
        return 'Sorry I do not understand.'

    res = anime.search(patterns, name)
    return 'Hope you enjoy watching these animes.'


def chatbot_response(msg, name=""):
    # ints = predict_class(msg, model)
    # res = getResponse(ints, intents)
    patterns = predict_class(msg, model)
    res = getResponse(patterns, name)
    return res


def chat():
    print("Find your favorite anime (type quit to stop)!")
    while True:
        inp = input("Search: ")
        if inp.lower() == "quit":
            break

        # extract in quotes and symbol ...
        inp_name = re.findall(r'"([^"]*)"', inp)
        # print (inp_name)

        if inp.strip() != '':
            if len(inp_name):
                results = chatbot_response(inp, inp_name[0])
            results = chatbot_response(inp)
            # print(results)


chat()
