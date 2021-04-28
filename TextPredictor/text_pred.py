import nltk
import os
# import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
# import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model
import pickle
# from jupyterthemes import jtplot

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Obtain the total words present in business dataset

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        result.append(token)
            
    return result

def train():
      df2 = pd.read_csv("train.csv")
      df2.drop(columns = ['ArticleId'], inplace = True)
      gk=df2.groupby('Category')
      business=gk.get_group('business')
      tech=gk.get_group("tech")
      sport=gk.get_group("sport")
      entertainment=gk.get_group("entertainment")
      politics=gk.get_group("politics")
      business['clean'] = business['Text'].apply(preprocess)
      list_of_words = []
      for i in business.clean:
          for j in i:
              list_of_words.append(j)

      length = 10 + 1
      lines = []

      for i in range(length, len(list_of_words)):
        seq = list_of_words[i-length:i]
        line = ' '.join(seq)
        lines.append(line)
        if i > 110000:
          break
      tokenizer = Tokenizer()
      tokenizer.fit_on_texts(lines)
      sequences = tokenizer.texts_to_sequences(lines)

      sequences = np.array(sequences)
      X, y = sequences[:, :-1], sequences[:,10]
      vocab_size = len(tokenizer.word_index) + 1
      y = to_categorical(y, num_classes=vocab_size)

      seq_length = X.shape[1]


      model = Sequential()
      model.add(Embedding(vocab_size, 10, input_length=seq_length))
      model.add(LSTM(100, return_sequences=True))
      model.add(LSTM(100))
      model.add(Dense(100, activation='relu'))
      model.add(Dense(vocab_size, activation='softmax'))


      model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
      model.fit(X, y, batch_size = 256, epochs = 100)

      model.save('text_prediction.model')



# def generate_text_seq(model, tokenizer, text_seq_length, seed_text, n_words):
#   text = []

#   for _ in range(n_words):
#     encoded = tokenizer.texts_to_sequences([seed_text])[0]
#     encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating='pre')

#     y_predict = model.predict_classes(encoded)

#     predicted_word = ''
#     for word, index in tokenizer.word_index.items():
#       if index == y_predict:
#         predicted_word = word
#         break
#     seed_text = seed_text + ' ' + predicted_word
#     text.append(predicted_word)
#   return ' '.join(text)

def Bgen(seed_text):
  dbfile = open('examplePickle', 'rb')     
  tokenizer = pickle.load(dbfile)
  text = []
  text_seq_length=10
  model = load_model('text_prediction.model')
  for _ in range(5):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating='pre')

    y_predict = model.predict_classes(encoded)

    predicted_word = ''
    for word, index in tokenizer.word_index.items():
      if index == y_predict:
        predicted_word = word
        break
    seed_text = seed_text + ' ' + predicted_word
    text.append(predicted_word)
  return ' '.join(text)

def Tgen(seed_text):
  dbfile = open('examplePickle(Tech)', 'rb')     
  tokenizer = pickle.load(dbfile)
  text = []
  text_seq_length=10
  model = load_model('Tech.model')
  for _ in range(5):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating='pre')

    y_predict = model.predict_classes(encoded)

    predicted_word = ''
    for word, index in tokenizer.word_index.items():
      if index == y_predict:
        predicted_word = word
        break
    seed_text = seed_text + ' ' + predicted_word
    text.append(predicted_word)
  return ' '.join(text)

def Sgen(seed_text):
  dbfile = open('examplePickle(Sport)', 'rb')     
  tokenizer = pickle.load(dbfile)
  text = []
  text_seq_length=10
  model = load_model('Sport.model')
  for _ in range(5):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating='pre')

    y_predict = model.predict_classes(encoded)

    predicted_word = ''
    for word, index in tokenizer.word_index.items():
      if index == y_predict:
        predicted_word = word
        break
    seed_text = seed_text + ' ' + predicted_word
    text.append(predicted_word)
  return ' '.join(text)

def Egen(seed_text):
  dbfile = open('examplePickle(Entertainment)', 'rb')     
  tokenizer = pickle.load(dbfile)
  text = []
  text_seq_length=10
  model = load_model('Entertainment.model')
  for _ in range(5):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating='pre')

    y_predict = model.predict_classes(encoded)

    predicted_word = ''
    for word, index in tokenizer.word_index.items():
      if index == y_predict:
        predicted_word = word
        break
    seed_text = seed_text + ' ' + predicted_word
    text.append(predicted_word)
  return ' '.join(text)

def Pgen(seed_text):
  dbfile = open('examplePickle(Politics)', 'rb')     
  tokenizer = pickle.load(dbfile)
  text = []
  text_seq_length=10
  model = load_model('Politics.model')
  for _ in range(5):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating='pre')

    y_predict = model.predict_classes(encoded)

    predicted_word = ''
    for word, index in tokenizer.word_index.items():
      if index == y_predict:
        predicted_word = word
        break
    seed_text = seed_text + ' ' + predicted_word
    text.append(predicted_word)
  return ' '.join(text)

