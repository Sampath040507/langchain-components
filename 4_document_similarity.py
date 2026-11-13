# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# emb2= OpenAIEmbeddings(model='text-embedding-3-large', dimensions= 32)
emb = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

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

print(cosine_similarity([query_emb], doc_emb))
#query_emb is a 384 dimensional vector so in array terms it is 1d but doc_emb is 2d, so we need to convert the 384 dimensional vector to 2d array i.e with 1 row and 384 columns where previously it wasn't. but when we need to compare-both should be same dimensional(For cosine similarity, each vector being compared must have the same number of dimensions). so me made query_emd 2d and doc_emb is already 2d.
#this result gives 2d result only but with 5 values since comparision of 1 with 5 gives only 5 i mean cosine similarity compares the one query vector against each of the five document vectors and returns a 1×5 similarity matrix(i.e. result is only 1 single vector in a list ...i.e u will have only 1 row in a 2d list i.e [[x1,x2,x3,x4,x5]]).

#NOTE:
# Q)confusion is how cosine similarity is checking only 1st vector of 2d list of left parameter with each and every vector in 2d list of right parameter?
# ANS:
# It is not specifically checking the "first vector" because cosine similarity has some special rule.
# It checks every vector in the left input against every vector in the right input.
# In your case, the left side simply contains only one vector.
#if 2 vectors in left and say left = [[l1], [l2]] then find similarity l1 with all r1-5 and l2 with all r1-5 .

#but we don't need 2d list , we need 1d list which is the first row only so [0]:
# print(cosine_similarity([query_emb], doc_emb)[0])

#NOTE 2:
# cosine_similarity() expects each argument to be a 2D collection of vectors.
# and sklearn's cosine_similarity() expects:
# (number of samples/vectors, number of features) for each argument or paramter     i.e (1,384), (5,384) but not (2,5,384) - 3d or more and not (384,) - 1d vector

scores = cosine_similarity([query_emb], doc_emb)[0]
#we need to sort the list in descending order so that we get the highest value resembling highest similarity but if we suffle without a plan, we can't find which similarity value belongs to which vector comparisions so we need to enumerate i.e assign with index so that if even shuffled or sorted we will still have a number which connects to the comparision of vectors i mean to the docs we wrote above:
#enumerate(list_name) gives an enumerate object not direct index, value pairs while printing so when you would like to print them u have to convert obj into list : list(enumerate(list_name))

#lambda- small anonymous fucntion used mostly with map, filter and sorted methods:

# sorted(list(enumerate(scores)), key= lambda x: x[1])
#this gives list in ascending order and key is the similarity score 
#now since sorted directly returns a list so based on index if u could retrieve required highest similarity value which is obv towards the end of list since list sorted in ascending order so sorted()[-1]:
# sorted(list(enumerate(scores)), key= lambda x: x[1])[-1]
#since returned value is a tuple with two indices let's store them in two differnet values:

index, score = sorted(list(enumerate(scores)), key= lambda x: x[1])[-1]

print(query)
print("information for ur query is:")
print(docs[index])
print()
print(f"similarity score:{score}")

