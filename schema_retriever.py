import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever


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
        self.vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 2. BM25 Retriever (Keyword Search)
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 3
        
    def get_relevant_schema(self, query):
        """
        Returns a formatted string containing ONLY the schemas relevant to the query.
        """
        # Fetch from both
        relevant_docs_vector = self.vector_retriever.invoke(query)
        relevant_docs_bm25 = self.bm25_retriever.invoke(query)
        
        # Merge uniquely to avoid duplicates
        unique_docs = {}
        for doc in relevant_docs_vector + relevant_docs_bm25:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
                
        # Combine the selected doc contents back into a string context
        combined_schema = "\n\n".join(unique_docs.keys())
        return combined_schema
