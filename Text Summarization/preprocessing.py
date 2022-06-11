# load libraries
# load necessary libraries
import numpy as np
import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize


# load data
def load_data(path):
    data = pd.read_csv(path)
    return data


# split text into sentences
def split_text(data):
    sentences = []
    for sentence in data['article_text']:
        sentences.append(sent_tokenize(sentence))


    sentences = [y for x in sentences for y in x]

    return sentences


# word embeding
def word_embedding(path):
    word_embeddings = {}

    file = open(path, encoding='utf-8')

    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs

    file.close()

    return word_embeddings

