import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def extract_insights():
    # Load preprocessed data
    df = pd.read_csv('data/preprocessed_data.csv')
    
    # Separate severe symptoms and medication side effects
    severe_symptoms = df[df['category'] == 'Emergency']['processed_content']
    side_effects = df[df['content'].str.contains('side effects', case=False)]['processed_content']
    
    # Function to extract top words from a topic
    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            print(f"Topic {topic_idx}: {', '.join(top_features)}")
    
    # Analyze severe symptoms
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(severe_symptoms)
    
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tfidf)
    
    print("Top words for severe symptoms topics:")
    print_top_words(lda, tfidf_vectorizer.get_feature_names_out(), 10)
    
    # Analyze medication side effects
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(side_effects)
    
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tfidf)
    
    print("\nTop words for medication side effects topics:")
    print_top_words(lda, tfidf_vectorizer.get_feature_names_out(), 10)

if __name__ == "__main__":
    extract_insights()
    print("NLP analysis completed.")