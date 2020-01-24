#!/usr/bin/env python
# coding: utf-8

# In[23]:


# Dense DNN for Movie Sentiment Analysis Part 1: Train
# 2020-01-21
# Jack Xue

from numpy import array
import string
import re
from os import listdir
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense


# In[24]:


# load doc into memory 
def load_doc(filename): 
    # open the file as read only 
    file = open(filename, 'r') 
    # read all text 
    text = file.read() 
    # close the file 
    file.close() 
    return text


# In[25]:


# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


# In[26]:


# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
    # load the doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)


# In[27]:


# load all docs in a directory
def process_docs(directory, vocab):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list
        lines.append(line)
    return lines


# In[28]:


# load and clean a dataset 
def load_clean_dataset(vocab): 
    # load documents 
    neg = process_docs('C:\\Users\\xinxue\\Desktop\\Family\\Roman\\review_polarity.tar\\review_polarity\\txt_sentoken\\neg', vocab) 
    pos = process_docs('C:\\Users\\xinxue\\Desktop\\Family\\Roman\\review_polarity.tar\\review_polarity\\txt_sentoken\\pos', vocab) 
    docs = neg + pos 
    # prepare labels 
    labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]) 
    return docs, labels


# In[29]:


# fit a tokenizer 
def create_tokenizer(lines): 
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(lines) 
    return tokenizer


# In[30]:


# integer encode and pad documents 
def encode_docs(tokenizer, max_length, docs): 
    # integer encode 
    ncoded = tokenizer.texts_to_sequences(docs) 
    # pad sequences 
    padded = pad_sequences(encoded, maxlen=max_length, padding='post') 
    return padded


# In[31]:


# define the model
def define_model(n_words, pngFile):
    # define network
    model = Sequential()
    model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    plot_model(model, to_file=pngFile, show_shapes=True)
    return model


# In[32]:


import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'
pngFile = 'C:\\Users\\xinxue\\Desktop\\Family\\Roman\\review_polarity.tar\\review_polarity\\txt_sentoken\\model.png'
#define_model(100,100,pngFile)


# In[33]:


# load the vocabulary
vocab_filename = 'C:\\Users\\xinxue\\Desktop\\Family\\Roman\\review_polarity.tar\\review_polarity\\txt_sentoken\\vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
# load all reviews
train_docs, ytrain = load_clean_dataset(vocab)
test_docs, ytest = load_clean_dataset(vocab)
# create the tokenizer
tokenizer = create_tokenizer(train_docs)
# encode data
Xtrain = tokenizer.texts_to_matrix(train_docs, mode='binary')
Xtest = tokenizer.texts_to_matrix(test_docs, mode='binary')
# define network
n_words = Xtrain.shape[1]
model = define_model(n_words, pngFile)
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)


# In[34]:


# classify a review as negative or positive
def predict_sentiment(review, vocab, tokenizer, model):
    # clean
    tokens = clean_doc(review)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    # convert to line
    line = ' '.join(tokens)
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode='tfidf')
    # predict sentiment
    yhat = model.predict(encoded, verbose=0)
    # retrieve predicted percentage and label
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return (1-percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'


# In[35]:


# test positive text
text = 'Best movie ever! It was great, I recommend it.'
percent, sentiment = predict_sentiment(text, vocab, tokenizer, model)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
# test negative text
text = 'This is a bad movie.'
percent, sentiment = predict_sentiment(text, vocab, tokenizer, model)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))


# In[ ]:




