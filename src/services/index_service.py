from src.services.auth_service import authentication_pinecone
from pinecone import ServerlessSpec

pc = authentication_pinecone()

def create_index(name: str):

    res = pc.create_index(
        name=name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

    return res

def list_index():

    res = pc.list_indexes()
    res = res.to_dict()
    return res


def detail_index(name: str):
    res = pc.describe_index(name=name)
    return res.to_dict()