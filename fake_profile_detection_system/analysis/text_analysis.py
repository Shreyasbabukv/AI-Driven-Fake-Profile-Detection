import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class TextAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def fit_transform(self, texts):
        """
        Fit the TF-IDF vectorizer and transform the texts.
        """
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        """
        Transform texts using the fitted TF-IDF vectorizer.
        """
        return self.vectorizer.transform(texts)

    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a text using TextBlob.
        Returns polarity and subjectivity.
        """
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
