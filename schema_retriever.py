import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

class SchemaRetrieverAgent:
    def __init__(self, schemas_list, openai_api_key=None):
        """
        Initializes the Ensemble Retriever (FAISS + BM25) for database schema strings.
        schemas_list: List of strings, where each string is a table's schema.
        """
        # Build Documents
        documents = [Document(page_content=schema) for schema in schemas_list]
        
        # Initialize Embeddings
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be provided for schema embeddings.")
            
        embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-3-small"
        )
        
        # Initialize Retrievers
        # 1. Vector Store Retriever (Semantic Search)
        vectorstore = FAISS.from_documents(documents, embeddings)
        # Search for top 3 matching tables by meaning
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 2. BM25 Retriever (Keyword Search)
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 3 # top 3 exact matches
        
        # Combine them using an EnsembleRetriever (weights out of 1.0)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        
    def get_relevant_schema(self, query):
        """
        Returns a formatted string containing ONLY the schemas relevant to the query.
        """
        relevant_docs = self.ensemble_retriever.invoke(query)
        # Combine the selected doc contents back into a string context
        combined_schema = "\n\n".join([doc.page_content for doc in relevant_docs])
        return combined_schema
