# load necessary libraries
import re
import string
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords   
import nltk

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

stop_words = stopwords.words('english')

# data path
path = "C:/Users/user/AppData/Roaming/nltk_data/corpora/twitter_samples/"

posetive_samples = twitter_samples.strings('C:/Users/user/AppData/Roaming/nltk_data/corpora/twitter_samples/positive_tweets.json')
negative_samples = twitter_samples.strings('C:/Users/user/AppData/Roaming/nltk_data/corpora/twitter_samples/negative_tweets.json')
text = twitter_samples.strings('C:/Users/user/AppData/Roaming/nltk_data/corpora/twitter_samples/tweets.20150430-223406.json')

tweet_tokens = twitter_samples.tokenized('C:/Users/user/AppData/Roaming/nltk_data/corpora/twitter_samples/positive_tweets.json')

print(tweet_tokens[0])

# create some functions to preprocess data
def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())

    return cleaned_tokens


# lets see our code is working untill now
print(remove_noise(tweet_tokens[0], stop_words))
