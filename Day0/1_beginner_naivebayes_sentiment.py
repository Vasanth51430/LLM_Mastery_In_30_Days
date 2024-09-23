import numpy as np
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

class SentimentAnalyzer:
    def __init__(self, max_features=5000, test_size=0.2, random_state=42):
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state
        self.pipeline = None
        self.stop_words = set(stopwords.words('english'))

    def load_data(self):
        """Load IMDB dataset"""
        print("Loading IMDB dataset...")
        dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
        train_dataset, test_dataset = dataset['train'], dataset['test']
        
        self.X = []
        self.y = []
        
        for text, label in train_dataset.concatenate(test_dataset):
            self.X.append(text.numpy().decode('utf-8'))
            self.y.append(label.numpy())
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        print(f"Dataset loaded. Shape: {self.X.shape}")

    def preprocess_text(self, text):
        """Clean and tokenize text"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)

    def prepare_data(self):
        """Preprocess and split the data"""
        print("Preprocessing data...")
        self.X = [self.preprocess_text(text) for text in self.X]
        
        print("Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

    def create_pipeline(self):
        """Create sklearn pipeline with TfidfVectorizer and MultinomialNB"""
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=self.max_features)),
            ('classifier', MultinomialNB())
        ])

    def train_model(self):
        """Train the model"""
        print("Training model...")
        self.create_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """Evaluate the model"""
        print("Evaluating model...")
        y_pred = self.pipeline.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred)}")

    def save_model(self, filepath):
        """Save the trained model"""
        joblib.dump(self.pipeline, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        self.pipeline = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

    def predict(self, texts):
        """Perform inference on new texts"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        predictions = self.pipeline.predict(processed_texts)
        return ['Positive' if pred == 1 else 'Negative' for pred in predictions]

def main():
    # Initialize SentimentAnalyzer
    analyzer = SentimentAnalyzer()

    # Load and prepare data
    analyzer.load_data()
    analyzer.prepare_data()

    # Train and evaluate model
    analyzer.train_model()
    analyzer.evaluate_model()

    # Save model
    analyzer.save_model('sentiment_model.joblib')

    # Load model (in a real scenario, this would be in a separate script)
    analyzer.load_model('sentiment_model.joblib')

    # Perform inference
    new_texts = [
        "This movie was excellent! The acting was superb.",
        "I didn't enjoy this film at all. The plot was confusing and the characters were poorly developed.",
        "An average movie. It had its moments but overall it was just okay."
    ]
    predictions = analyzer.predict(new_texts)
    
    print("\nNew predictions:")
    for text, prediction in zip(new_texts, predictions):
        print(f"Text: '{text}'\nPredicted sentiment: {prediction}\n")

if __name__ == "__main__":
    main()