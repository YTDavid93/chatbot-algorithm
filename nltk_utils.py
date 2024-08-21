import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def preprocess(text):
    lemmitizer = WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatized_tokens = [lemmitizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)
