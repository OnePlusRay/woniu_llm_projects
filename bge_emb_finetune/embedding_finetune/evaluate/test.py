from sentence_transformers import SentenceTransformer

# sentences ="数据1"
# model = SentenceTransformer('../../acge_text_embedding')
# print(model.max_seq_length)
# embeddings_1 = model.encode(sentences, normalize_embeddings=True)
# embeddings_2 = model.encode(sentences, normalize_embeddings=True)
# print(embeddings_1)
# print(type(embeddings_1))
# similarity = embeddings_1 @ embeddings_2.T
# print(similarity)


import os
import json
import sys
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import ElasticsearchStore
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from elasticsearch import Elasticsearch


current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, '..', 'utils')
sys.path.append(utils_dir)
from ACGE import ACGEEmbedding

embeddings = ACGEEmbedding()
a1 = embeddings.embed_query('你好')
a2 = embeddings.embed_documents(['你好','傻瓜'])

print(a1)
print(a2)