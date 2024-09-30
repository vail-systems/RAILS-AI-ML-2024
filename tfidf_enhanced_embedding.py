from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from transformers import BertTokenizer, BertModel
import torch

# You can change the embedding model to one available in the SentenceTransformers library or one available locally
embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
def preprocess(text):
    return text.lower()

def normalize(vector):
    vector = np.array(vector)
    norm = np.linalg.norm(vector)
    return vector / norm

def initialize(docs, n_components=1024):
    """
    Initialize the TF-IDF enhanced embedding process.

    This function preprocesses the input documents, computes their TF-IDF vectors,
    and then reduces the dimensionality of these vectors using Truncated SVD.

    Args:
        docs (list of str): A list of documents to be processed.
        n_components (int, optional): The number of components for Truncated SVD. Default is 1024.

    Returns:
        tuple: A tuple containing:
            - reduced_tfidf_matrix (numpy.ndarray): The reduced dimensionality TF-IDF matrix.
            - svd (TruncatedSVD): The fitted TruncatedSVD instance.
            - tfidf_vectorizer (TfidfVectorizer): The fitted TfidfVectorizer instance.
    """
    docuements_for_tfidf = [preprocess(doc) for doc in docs]
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(docuements_for_tfidf)
    # Reduce the dimensionality of TF-IDF vectors to 1024
    svd = TruncatedSVD(n_components=n_components)
    reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)

    return reduced_tfidf_matrix, svd, tfidf_vectorizer

# Reduce the dimensionality of TF-IDF vectors to 1024
def get_vector(query_str, svd, tfidf_vectorizer, embed_model):
    """
    Generate a combined vector representation of a query string using both SVD-transformed TF-IDF and embedding model.

    Args:
        query_str (str): The input query string to be vectorized.
        svd (sklearn.decomposition.TruncatedSVD): The SVD model used to reduce the dimensionality of the TF-IDF vectors.
        tfidf_vectorizer (sklearn.feature_extraction.text.TfidfVectorizer): The TF-IDF vectorizer used to transform the query string.
        embed_model (SentenceTransformer or similar): The embedding model used to generate embeddings for the query string.

    Returns:
        list: A combined vector representation of the query string, averaged from normalized TF-IDF and embedding vectors.
    """
    query_bge_embedding = embed_model.encode(query_str)
    query_tf_idf_embedding = svd.transform(tfidf_vectorizer.transform([preprocess(query_str)]))[0]
    normalized_query_bge_embedding = normalize(query_bge_embedding)
    normalized_query_tfidf_embedding = normalize(query_tf_idf_embedding)
    combined_query_embedding = list((normalized_query_bge_embedding + normalized_query_tfidf_embedding) / 2)

    return combined_query_embedding

def main():
    # Example documents
    # You'll probably need to replace these with your own documents.
    # BGE embedding returns a 1024-dimensional vector, so the TF-IDF vectors should also be at least 1024-dimensional for SVD to work.
    documents = [
        "This is an example document about machine learning.",
        "Another document is about natural language processing.",
        "A third document discusses computer vision."
    ]
    # Initialize the TF-IDF enhanced embedding process
    reduced_tfidf_matrix, svd, tfidf_vectorizer = initialize(documents)
    # Example query string
    query = "machine learning"
    # Generate a combined vector representation of the query string
    query_vector = get_vector(query, svd, tfidf_vectorizer, embed_model)
    return query_vector