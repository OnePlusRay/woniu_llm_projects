from typing import List
import requests
import json
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class ACGEEmbedding(Embeddings):
    def __init__(self,model_path:str='../../acge_text_embedding'):
        super().__init__()
        self.model_path = model_path

    def embed_query(self, text: str) -> List[float]:
        model = SentenceTransformer(self.model_path)
        embedding = model.encode(text, normalize_embeddings=True).tolist()
        return embedding
    
    def embed_documents(self, texts:List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
