
<H1> OBJECTIVE </H1>
  
This project is focused on emulation of text classification tool for bulk emails and text messages that OSF Healthcare receives. Due to it being a healthcare organization, OSF receives huge amount of queries from patients, their families, general citizens and also spams. Naturally, siphoning manually through emails and responding to emergency messages is not only inefficient but a delayed response might even lead to severe consequences such as untimely advice about medication dosage or side effects monitoring. 

![objectiveImage](https://github.com/user-attachments/assets/7c8ed120-eeb6-4ae2-825b-b3d008e0d6c1)

This classification tool aims to not only reduce the time to reduce the amount of time taken to identify important messages but gives reponders room to answer the questions from patients or their family members with higher importance to priority messages.



<H1> APPROACH </H1>

1. We start with dummy data creation using a python script
   
![DataGeneration](https://github.com/user-attachments/assets/659d31cc-03fc-4405-bbf4-b83c77182c8a)

This script creates a dataframe consisting features such as timestamps, content of the texts and emails.

2. Preprocessing the text data, removing the stopwords, dealing with special characters, removing capatalized letters and at the end combining the text and email dataframes together ready for model prediction.

![preprocessing](https://github.com/user-attachments/assets/ca0ba559-5beb-44cf-becd-3d7f5c1d4d39)

3. Before using the preprocessed data, we are gonna perform a common NLP technique such as TF/IDF to identify important terms in the messages.

   How TF/IDF works:

  # TF and TF-IDF in NLP

  ## Term Frequency (TF)

**Definition:** 
Term Frequency measures how frequently a term appears in a document. The formula for calculating TF for a term \( t \) in a document \( d \) is:

TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)


**Example:**
Consider a document with the following text:

"The cat sat on the mat. The cat is happy."


- Total number of terms = 10 (including repetitions)
- "cat" appears 2 times.

TF(cat, d) = 2 / 10 = 0.2

## Term Frequency-Inverse Document Frequency (TF-IDF)

**Definition:**
TF-IDF combines TF with Inverse Document Frequency (IDF) to evaluate how important a word is in a document relative to a collection of documents. The IDF for a term \( t \) is calculated as follows:

IDF(t) = log(Total number of documents / Number of documents containing t)


Then, the TF-IDF score is calculated as:

TFIDF(t, d) = TF(t, d) * IDF(t)


**Example:**
Suppose we have three documents:

1. Document 1: "The cat sat on the mat."
2. Document 2: "The dog barks at the cat."
3. Document 3: "Cats are great pets."

### Step 1: Calculate TF for "cat" in each document.

- **Doc 1:** 1/7 = 0.14 (7 total words)
- **Doc 2:** 1/7 = 0.14
- **Doc 3:** 1/5 = 0.2

### Step 2: Calculate IDF for "cat."

- Total documents = 3
- Documents containing "cat" = 2


​
**Example:**
Suppose we have three documents:

1. Document 1: "The cat sat on the mat."
2. Document 2: "The dog barks at the cat."
3. Document 3: "Cats are great pets."

### Step 1: Calculate TF for "cat" in each document.

- **Doc 1:** 1/7 = 0.14 (7 total words)
- **Doc 2:** 1/7 = 0.14
- **Doc 3:** 1/5 = 0.2

### Step 2: Calculate IDF for "cat."

- Total documents = 3
- Documents containing "cat" = 2

IDF(cat) = log(3 / 2) ≈ 0.176


### Step 3: Calculate TF-IDF for "cat" in each document.

- **Doc 1:** 
TFIDF(cat, Doc 1) = 0.14 * 0.176 ≈ 0.025

- **Doc 2:** 
TFIDF(cat, Doc 2) = 0.14 * 0.176 ≈ 0.025

- **Doc 3:** 
TFIDF(cat, Doc 3) = 0.2 * 0.176 ≈ 0.035

![wordcloud](https://github.com/user-attachments/assets/80daff4a-f74e-4c77-81d1-1ad0b0c434c5)


    
3. Approach
  1. Script description
  2. Results
  3. Images
4. Overall Result
5. Future Plans
