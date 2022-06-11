# load libraries
# loadd libraries
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics,svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, numpy, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers


from preprocessing import load_data, count_vector


# load data
path = "G:/rauf/STEPBYSTEP/Data/amazon_reviews.txt"

x_train, x_test, y_train, y_test, dataframe = load_data(path)

xtrain_count, xtest_count = count_vector(dataframe, x_train, x_test)

# train the model
def train_model(classifier, x_train, y_train, x_test, y_test, is_neural_net = False):
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    if is_neural_net:
        y_pred = y_pred.argmax(axis=-1)

    return metrics.accuracy_score(y_pred, y_test)

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, y_train, xtest_count, y_test)
print(accuracy)

# what a fuck
# it worked 
# unbeliable
# with accuracy 70%
# rauf odilov