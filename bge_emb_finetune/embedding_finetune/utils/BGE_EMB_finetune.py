from typing import List
import requests
import json
from FlagEmbedding import FlagModel
from langchain_core.embeddings import Embeddings

class BGEEmbedding_finetune(Embeddings):
    def __init__(self,model_path:str,query_instruction_for_retrieval:str='',use_fp16:bool=True):
        super().__init__()
        self.model_path = model_path
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.use_fp16 = use_fp16
        self.model = FlagModel(model_name_or_path=self.model_path,
                          query_instruction_for_retrieval=self.query_instruction_for_retrieval,
                          use_fp16=self.use_fp16
                          )

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode_queries(text).tolist()
        
        return embedding
    
    def embed_documents(self, texts:List[str]) -> List[List[float]]:
        embeddings = self.model.encode_corpus(texts).tolist()
        return embeddings
    