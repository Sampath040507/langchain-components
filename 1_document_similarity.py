from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

emb= OpenAIEmbeddings(model='text-embedding-3-large', dimensions= 32)
# emb = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

docs = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about rohit"

doc_emb = emb.embed_documents(docs)
query_emb = emb.embed_query(query)

# print(cosine_similarity([query_emb], doc_emb))

scores = cosine_similarity([query_emb], doc_emb)[0]
index, score = sorted(list(enumerate(scores)), key= lambda x: x[1])[-1]

print(query)
print("information for ur query is:")
print(docs[index])
print()
print(f"similarity score:{score}")
