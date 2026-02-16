import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

# Inisialisasi stemmer
stemmer = LancasterStemmer()

# Load data intents
intents = json.loads(open('intents.json').read())

# Inisialisasi list kata-kata
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Proses data intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenisasi kata-kata
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Tambahkan ke dokumen
        documents.append((w, intent['tag']))
        # Tambahkan ke kelas
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Stemming kata-kata
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Simpan kata-kata dan kelas
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Inisialisasi training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # Inisialisasi bag of words
    bag = []
    # List kata-kata
    pattern_words = doc[0]
    # Stemming kata-kata
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # Buat bag of words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output adalah '0' untuk setiap tag dan '1' untuk tag saat ini
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle training data
random.shuffle(training)
training = np.array(training)

# Buat train dan test sets
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Buat model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0],),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Simpan model
model.save('chatbot_model.h5', hist)

print("Model selesai dilatih")