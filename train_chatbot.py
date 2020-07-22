import random
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
import numpy as np
import pickle
import json
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


words = []
words_with_tags = []
classes = []
documents = []
ignore_words = ['?', '!', '-']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        words_with_tags.append({
            'pattern': w[0],
            'tag': intent['tag']
        })

        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(w.lower())
         for w in words if w not in ignore_words]

words = list(words)

classes = sorted(list(set(classes)))

print(len(documents), "documents")

print(len(classes), "classes", classes)

print(len(words), "unique lemmatized words", words)

# convert words into byte stream file
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
pickle.dump(words_with_tags, open('words_with_tags.pkl', 'wb'))

# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(
        word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax

# | Number of Hidden Layers | Result |
#  0 - Only capable of representing linear separable functions or decisions.
#  1 - Can approximate any function that contains a continuous mapping
# from one finite space to another.
#  2 - Can represent an arbitrary decision boundary to arbitrary accuracy
# with rational activation functions and can approximate any smooth
# mapping to any accuracy.

# A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input and one output.
model = Sequential()
# Create a layer with 128 neurons
# Dense is just a word for fully-connected neural network
# Fully connected neural networks (FCNNs) are a type of artificial neural network where the architecture is such that all the nodes, or neurons, in one layer are connected to the neurons in the next layer.
# input_shape takes in the tuple that represents dimension of the layer. In this case, it is just one-dimensional
# len(train_x[0]) is the number of patterns // 51
# activation is a function that defines how the output should be, given the input
# activation relu is rectified linear unit activation, it works like max(0, x)
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# Dropout is used to ignore some nodes which means when we create 128 neurons/node, we only selected half of it so that the selected nodes are more spreaded out
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# activation softmax returns the probability on the interval(0,1)
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model
# epoch = how many times you go through your training set.
# batch size = number of samples it selectes in each training
# verbose just show progress bar when training
hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")
