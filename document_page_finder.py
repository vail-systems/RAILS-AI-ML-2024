from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llama_index import SimpleDirectoryReader

def read_original_pdfs(path):
    file_metadata = lambda x : {"filename": x}
    reader = SimpleDirectoryReader(path, file_metadata=file_metadata, recursive=True, required_exts=['.pdf'])
    documents = reader.load_data()
    return documents

def get_pdf_pages_texts(desired_filename, documents):
    """
    Get the text of all pages of a document with the desired filename.

    Args:
        desired_filename (str): The desired filename.
        documents (list): A list of document objects.

    Returns:
        dict: A dictionary where the keys are the page numbers and the values are the text of the pages.
    """
    pages_texts = {}
    for doc in documents:
        if doc.metadata['filename'].split('/')[-1].split('.pdf')[0] == desired_filename:
            pages_texts[doc.metadata['page_label']] = doc.text
    return pages_texts

# Example data
def search_page(pages_dict, query):
    """
    Search for the query in the pages dictionary and return the page number with the highest similarity.

    Args:
        pages_dict (dict): A dictionary where the keys are the page numbers and the values are the text of the pages.
        query (str): The query to search for.

    Returns:
        str: The page number with the highest similarity to the query.
    """
    vectorizer = TfidfVectorizer()
    tfidf_pages = vectorizer.fit_transform(list(pages_dict.values()))
    tfidf_query = vectorizer.transform([query])
    # print(tfidf_query.shape, tfidf_pages.shape)
    similarities = cosine_similarity(tfidf_query, tfidf_pages)
    best_match = similarities.argmax()
    return list(pages_dict.keys())[best_match]

def get_node_page_dict(response, documents_original_pdf):
    """
    Returns a dictionary mapping source node indices to corresponding page numbers in a PDF document.

    Args:
        response (Response): The response object containing source nodes.
        documents_original_pdf (str): The path to the original PDF document.

    Returns:
        dict: A dictionary mapping source node indices to a tuple containing the desired filename and page number.
              If the original document is not a PDF, it is set to None.
    """
    pages_chunk_dict = {}
    for i in range(len(response.source_nodes)):
        desired_filename = response.source_nodes[i].metadata['filename'].split('/')[-1].split('.txt')[0]
        # print(desired_filename)
        pages_dict = get_pdf_pages_texts(desired_filename, documents_original_pdf)
        if pages_dict:
            page_number = search_page(pages_dict, response.source_nodes[i].text)
            pages_chunk_dict[f"sourcenode_{i}"] = (desired_filename, page_number)
        else:
            pages_chunk_dict[f"sourcenode_{i}"] = (desired_filename, None)
    return pages_chunk_dict

def main():
    # Read original PDFs
    documents_original_pdf = read_original_pdfs('/path/to/your/data')
    # Example response object
    response = {
        "source_nodes": [
            {
                "text": "This is an example text from the source node 1.",
                "metadata": {
                    "filename": "/path/to/source_node_1.txt"
                }
            },
            {
                "text": "This is an example text from the source node 2.",
                "metadata": {
                    "filename": "/path/to/source_node_2.txt"
                }
            }
        ]
    }
    npd = get_node_page_dict(response, documents_original_pdf)
    return npd