import numpy as np
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

def Vectorize(sentences):
    sentence_embeddings = sbert_model.encode(sentences)
    # np.save('Processed/vect_data', sentence_embeddings)
    print("Loading")
    vect_data = np.load('Processed/vect_data.npy', allow_pickle=True)
    return vect_data