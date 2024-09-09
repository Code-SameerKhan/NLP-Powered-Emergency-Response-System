import pandas as pd
import random
from datetime import datetime, timedelta

def generate_dummy_data(num_samples=1000):
    categories = ['Emergency', 'Urgent', 'Non-urgent', 'Inquiry']
    symptoms = ['Fever', 'Cough', 'Shortness of breath', 'Chest pain', 'Headache', 'Nausea']
    medications = ['Aspirin', 'Ibuprofen', 'Acetaminophen', 'Amoxicillin', 'Lisinopril']

    data = []
    for _ in range(num_samples):
        category = random.choice(categories)
        symptom = random.choice(symptoms)
        medication = random.choice(medications)
        
        if category == 'Emergency':
            content = f"URGENT: Patient experiencing severe {symptom}. Immediate assistance required."
        elif category == 'Urgent':
            content = f"Patient reporting persistent {symptom}. Requires prompt attention."
        elif category == 'Non-urgent':
            content = f"Follow-up needed for mild {symptom}."
        else:
            content = f"Question about {medication} side effects."

        data.append({
            'timestamp': datetime.now() - timedelta(minutes=random.randint(0, 1000)),
            'content': content,
            'category': category
        })

    df = pd.DataFrame(data)
    return df

def generate_and_save_data():
    emails_df = generate_dummy_data(800)  # Default of 800 emails
    texts_df = generate_dummy_data(200)   # Default of 200 texts

    # Save the data to CSV
    emails_df.to_csv('./data/dummy_emails.csv', index=False)
    texts_df.to_csv('./data/dummy_texts.csv', index=False)

    print("Dummy data generated and saved to CSV files.")


if __name__ == "__main__":
    generate_and_save_data()