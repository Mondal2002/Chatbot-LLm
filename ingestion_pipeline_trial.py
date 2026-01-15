# from sentence_transformers import SentenceTransformer

# # Load a pre-trained embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# text = "ChatNex is an AI-powered frontdesk "

# embedding = model.encode(text)

# print(embedding)  # 384 dimensions

import pandas as pd

documents =pd.read_csv('Todung_knowledgebase.txt',sep='\t')

print(documents)
