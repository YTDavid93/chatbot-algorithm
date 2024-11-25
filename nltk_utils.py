import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def preprocess(text):
    stemmer = PorterStemmer()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and lowercase
    tokens = nltk.word_tokenize(text.lower())
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Apply stemming
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)