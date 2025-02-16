import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# Example preprocessing function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text=text.lower()
    text=re.sub(r'\W',' ',text)
    text=' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text

