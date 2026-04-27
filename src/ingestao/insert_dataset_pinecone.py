import argparse
import pandas as pd
import logging
from ..utils import gerar_hash_md5, length_function_tkt, carregar_dataset
from ..services.index_service import list_index, create_index

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)

def transform_dataframe_in_documents(df):
    docs = []
    for _, row in df.iterrows():
        if row['data_category_QA'] in ['positivo', 'negativo']:
            doc = Document(
                    page_content = row['content'],
                    metadata = {'data_category_QA': row['data_category_QA'],
                                'md5': row['md5']
                            }
                )
            
            docs.append(doc)
            # break
    
    return docs

def gerar_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 100,
        length_function = length_function_tkt,
        separators = ["\n\n", "\n", ".", " ", ""]
    )

    # Gerando chunks e mantendo metadados do documento pai
    chunks = text_splitter.split_documents(documents)

    return chunks

def inserir_naive(chunks, embeddings, index_name):
    logging.info('Inserindo chunks no banco vetorial')
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )


def inserir_em_batches(chunks, embeddings, index_name, batch_size=100, start_batch=0):
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    total = len(chunks)
    batches = range(start_batch * batch_size, total, batch_size)

    for i in tqdm(batches, desc='Inserindo chunks', unit='batch', initial=start_batch, total=total // batch_size + 1):
        batch = chunks[i:i + batch_size]
        ids = [f"{chunk.metadata['md5']}_{i + j}" for j, chunk in enumerate(batch)]
        vector_store.add_documents(batch, ids=ids)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('index_name', type=str, help='Nome do index no Pinecone')
    parser.add_argument('--start-batch', type=int, default=0, help='Batch para retomar a inserção (default: 0)')
    args = parser.parse_args()

    index_name = args.index_name
    start_batch = args.start_batch

    # Criando index no Pinecone
    indexes = list_index()
    indexes = [ind['name'] for ind in indexes['indexes']]
    if index_name not in indexes:
        create_index(name=index_name)
        logging.info('Index criado com sucesso!')
    else:
        logging.info('Index já existe! Verifique sua conta administradora no Pinecone.')
        exit()
    
    df = carregar_dataset()

    df['md5'] = df.content.apply(gerar_hash_md5)
    documents = transform_dataframe_in_documents(df)
    logging.info(f'Total de documents: {len(documents)}')

    logging.info('Gerando chunks...')
    chunks = gerar_chunks(documents)
    logging.info(f'Total de chunks gerados: {len(chunks)}')

    logging.info('Definindo modelo de embeddings...')
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    logging.info('Inserindo chunks no banco vetorial...')
    inserir_em_batches(chunks, embeddings, index_name, start_batch=start_batch)
    logging.info('Dados inseridos com sucesso!')
    

if __name__ == '__main__':
    main()