import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


from dataGenerationScript import generate_and_save_data
from preprocessingScript import preprocess_and_save
from classificationModelScript import train_model
from NLPAnalysisScript import extract_insights
from visualizationScript import create_visualizations

def main():
    # Generate fake medical emails and texts
    print("Generating dummy data...")
    generate_and_save_data()
    print()
    
    # Preprocessing data i.e., removing stopwords, casing etc.
    print("Preprocessing data...")
    preprocess_and_save()
    print()
    
    # Training a Classification model
    print("Training classification model...")
    train_model()
    print()
    
    # Find top 10 Symptoms and Medication Side effects
    print("Performing NLP analysis...")
    extract_insights()
    print()
    
    # Create visualizations like word clouds and bar graphs
    print("Creating visualizations...")
    create_visualizations()
    print()
    print("Emergency Response System simulation completed.")

if __name__ == "__main__":
    main()