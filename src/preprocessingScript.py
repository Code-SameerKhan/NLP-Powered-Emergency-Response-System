import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['processed_content'] = df['content'].apply(preprocess_text)
    return df

# Preprocess emails and texts
emails_df = preprocess_data('../data/dummy_emails.csv')
texts_df = preprocess_data('../data/dummy_texts.csv')

# Combine datasets
combined_df = pd.concat([emails_df, texts_df], ignore_index=True)

# Save preprocessed data
combined_df.to_csv('../data/preprocessed_data.csv', index=False)

print("Data preprocessing completed.")