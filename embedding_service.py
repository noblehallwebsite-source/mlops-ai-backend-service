from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity



model = SentenceTransformer(

    "sentence-transformers/all-MiniLM-L6-v2"

)



def generate_embedding(text: str):

    embedding = model.encode(text)

    return embedding.tolist()





def calculate_similarity(text1: str, text2: str):



    embeddings = model.encode([text1, text2])



    similarity = cosine_similarity(

        [embeddings[0]],

        [embeddings[1]]

    )[0][0]



    return float(similarity)  