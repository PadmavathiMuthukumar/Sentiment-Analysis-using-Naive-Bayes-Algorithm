# Sentiment-Analysis-using-Naive-Bayes-Algorithm

## AIM:
To develop a simple bilingual translation model that translates English sentences to French using TF-IDF vectorization and K-Nearest Neighbors (KNN) based on cosine similarity.

## THEORY:
Bilingual translation is the task of mapping a sentence in one language to its equivalent in another language. In this implementation, TF-IDF (Term Frequency-Inverse Document Frequency) is used to vectorize text, and cosine similarity is used to measure the closeness of sentences.

**Key Concepts:**
TF-IDF Vectorizer: Converts textual data into numerical form based on word importance in the dataset.

Cosine Similarity: Measures similarity between two vectors by comparing their direction.

KNN for Translation: Finds the most similar English sentence in a bilingual dictionary and returns the corresponding French sentence.

## PROCEDURE:
STEP . 1 : Import Required Libraries: Use sklearn for vectorization and similarity measurement.

STEP . 2 : Prepare a Bilingual Dataset: Create a parallel list of English and French sentence pairs.

STEP . 3 : Vectorize English Sentences: Use TfidfVectorizer to encode English sentences into numerical vectors.

STEP . 4 : Define KNN Translation Function:

STEP . 5 : Convert input sentence to TF-IDF vector.

STEP . 6 : Calculate cosine similarity with all English sentences in the dataset.

STEP . 7 : Find the top-k most similar sentences and return corresponding French translations.

STEP . 8 : Test the Translator: Provide English input and return predicted French translations.

## PROGRAM:
``` python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample bilingual dictionary
english_sentences = [
    "hello", "how are you", "good morning", "good night", "thank you",
    "see you later", "what is your name", "my name is John", "where is the library",
    "I like to read books"
]

french_sentences = [
    "bonjour", "comment ça va", "bonjour", "bonne nuit", "merci",
    "à plus tard", "quel est ton nom", "mon nom est John", "où est la bibliothèque",
    "j'aime lire des livres"
]

# Vectorize the English sentences
vectorizer = TfidfVectorizer()
english_vectors = vectorizer.fit_transform(english_sentences)

# Define translation function using KNN
def knn_translate(input_sentence, k=1):
    input_vector = vectorizer.transform([input_sentence])
    similarities = cosine_similarity(input_vector, english_vectors).flatten()
    top_k_indices = similarities.argsort()[-k:][::-1]
    translations = [french_sentences[i] for i in top_k_indices]
    return translations

# Test sentences
test_sentences = ["good evening", "where is the library", "thank you very much"]

# Translate and display results
for sentence in test_sentences:
    translations = knn_translate(sentence, k=1)
    print(f"English: {sentence} -> French: {translations[0]}")
```
## OUTPUT:
![Screenshot 2025-06-04 191030](https://github.com/user-attachments/assets/230c69c3-aad2-42eb-ba81-18dfb3f9dbb0)

## RESULT:
The bilingual translation model was successfully implemented using TF-IDF vectorization and cosine similarity-based KNN.
