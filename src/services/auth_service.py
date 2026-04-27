from pinecone.grpc import PineconeGRPC as Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

def authentication_pinecone():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc