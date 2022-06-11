# loadd libraries
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics,svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, numpy, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers


# load dataset

def load_data(path):
    data = open(path).read()
    labels, texts = [], []

    for i, line in enumerate(data.split("\n")):
        content = line.split()
        labels.append(content[0])
        texts.append("".join(content[1:]))

    dataframe = pandas.DataFrame()
    dataframe['texts'] = texts
    dataframe['labels'] = labels

    # split data
    train_x, validate_x, train_y, validate_y = model_selection.train_test_split(dataframe['texts'], dataframe['labels'], test_size = 0.2)

    # encode labels
    encoder = preprocessing.LabelEncoder()

    train_y = encoder.fit_transform(train_y) 
    validate_y = encoder.fit_transform(validate_y)

    return train_x, validate_x, train_y, validate_y, dataframe


# feature enginering


# Count vectors
def count_vector(train_data, x_train, x_valid):
    count_vector = CountVectorizer(analyzer='word', token_pattern = r'\w{1,}')
    count_vector.fit(train_data['texts'])

    xtrain_count = count_vector.transform(x_train)
    xvalid_count = count_vector.transform(x_valid)

    return xtrain_count, xvalid_count


# we will use embeddings as feature
def embedding(path, train_data):
    embeddings_index = {}
    for i, line in enumerate(open(path)):
        values = line.split()

        embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

        # create tokenizer
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(train_data['texts'])
        word_index = tokenizer.word_index


    return None

