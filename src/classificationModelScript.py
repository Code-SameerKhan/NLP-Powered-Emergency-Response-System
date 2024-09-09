import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

def train_model():
    # Load preprocessed data
    df = pd.read_csv('data/preprocessed_data.csv')
    
    # Prepare features and target
    X = df['processed_content']
    y = df['category']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save the pipeline
    joblib.dump(pipeline, 'src/models/classifier_pipeline.joblib')

if __name__ == "__main__":
    train_model()
    print("Model trained and saved.")