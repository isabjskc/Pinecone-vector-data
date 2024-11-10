# get_embeddings.py
import sys
import json
from sentence_transformers import SentenceTransformer

def get_embeddings(texts):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings.tolist()

if __name__ == "__main__":
    # Read input from command line
    input_text = sys.argv[1]
    texts = json.loads(input_text)
    embeddings = get_embeddings(texts)
    print(json.dumps(embeddings))
