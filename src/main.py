import argparse
import csv
import json
import pandas as pd
import os
import logging
from datetime import datetime
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError

from sklearn.metrics import classification_report

from langchain_openai import OpenAI, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

from .prompts.prompts_template import template, template_com_justificativa
from .utils import carregar_dataset, gerar_hash_md5, parse_llm_json

logging.basicConfig(level=logging.INFO)

def get_output_file(model_name: str):
    folder = f'src/data/{model_name}'
    os.makedirs(folder, exist_ok=True)
    return f'{folder}/resultados.jsonl'

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def invoke_com_retry(invocavel, input):
    return invocavel.invoke(input)


def carregar_processados(output_file: str):
    processados = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                registro = json.loads(line)
                processados.add((registro['md5'], registro['question'], registro['tipo']))
    return processados

def _garantir_labels(df: pd.DataFrame) -> pd.DataFrame:
    if 'data_category_QA' in df.columns and df['data_category_QA'].notna().any():
        return df
    logging.info('data_category_QA ausente nos resultados — fazendo merge com o dataset original.')
    df_original = carregar_dataset()
    df_original['md5'] = df_original['content'].apply(gerar_hash_md5)
    labels = df_original[['md5', 'question', 'data_category_QA']].drop_duplicates(subset=['md5', 'question'])
    return df.drop(columns=['data_category_QA'], errors='ignore').merge(labels, on=['md5', 'question'], how='left')


def show_metrics(df: pd.DataFrame):
    df = _garantir_labels(df)
    metrics = {}

    df_rag = df[df.tipo == 'rag']
    df_rag = df_rag[~df_rag.duplicated(subset=['md5', 'question'])]
    print()
    print('===== Results =====')
    print('Resultado utilizando chunks mais relevantes')
    print('Shape dataset tipo RAG:', df_rag.shape)
    print(classification_report(df_rag.data_category_QA, df_rag.pred))
    metrics['rag'] = classification_report(df_rag.data_category_QA, df_rag.pred, output_dict=True)
    print()

    df_full_context = df[df.tipo == 'full_context']
    df_full_context = df_full_context[~df_full_context.duplicated(subset=['md5', 'question'])]
    print('Resultado utilizando todo o documento')
    print('Shape dataset tipo Full Context:', df_full_context.shape)
    print(classification_report(df_full_context.data_category_QA, df_full_context.pred))
    metrics['full_context'] = classification_report(df_full_context.data_category_QA, df_full_context.pred, output_dict=True)
    print()

    return metrics


EXPERIMENTS_FILE = 'experiments.csv'
EXPERIMENTS_FIELDS = ['data', 'modelo', 'tipo', 'accuracy', 'f1_macro', 'f1_positivo',
                      'recall_positivo', 'precision_positivo', 'n_amostras', 'n_chunks']

def registrar_experimento(model_name: str, metrics: dict, n_chunks: int):
    file_exists = os.path.exists(EXPERIMENTS_FILE)

    rows = []
    for tipo, report in metrics.items():
        if not report:
            continue
        pos = report.get('positivo', {})
        rows.append({
            'data': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'modelo': model_name,
            'tipo': tipo,
            'accuracy': round(report.get('accuracy', 0), 4),
            'f1_macro': round(report['macro avg']['f1-score'], 4),
            'f1_positivo': round(pos.get('f1-score', 0), 4),
            'recall_positivo': round(pos.get('recall', 0), 4),
            'precision_positivo': round(pos.get('precision', 0), 4),
            'n_amostras': int(report['macro avg']['support']),
            'n_chunks': n_chunks if tipo == 'rag' else 'N/A',
        })

    with open(EXPERIMENTS_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENTS_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    logging.info(f'Experimento registrado em {EXPERIMENTS_FILE}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('index_name', type=str, help='Nome do index no Pinecone')
    parser.add_argument('--model-name', type=str, default='gpt-5.4-nano', help='Modelo OpenAI (Chat Completions) a utilizar')
    args = parser.parse_args()

    model_name = args.model_name
    index_name = args.index_name
    output_file = get_output_file(model_name)

    df = carregar_dataset()
    df = df[(df['data_category_QA']=='positivo') | (df['data_category_QA']=='negativo')]
    logging.info(f'dataset filtrado: {df.shape}')
    df['md5'] = df.content.apply(gerar_hash_md5)

    processados = carregar_processados(output_file)
    logging.info(f'{len(processados)} itens já processados, retomando...')

    llm = ChatOpenAI(model_name=model_name).bind(
        response_format={"type": "json_object"}
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    NUM_CHUNKS = 3

    with open(output_file, 'a') as f:

        # Avaliação RAG - chunks mais relevantes
        pendentes_rag = df[~df.apply(lambda r: (r['md5'], r['question'], 'rag') in processados, axis=1)]
        logging.info(f'{len(pendentes_rag)} questões pendentes para avaliação RAG')

        for _, row in tqdm(pendentes_rag.iterrows(), total=len(pendentes_rag), desc='RAG', unit='q'):
            md5 = row['md5']

            # Limitei a pesquisa vetorial ao md5, pois é importante isolar os documentos que apenas interessa.
            # Deixar de limitar também pelo md5, a qualidade de recuperação vetorial pode comprometer o experimento
            # O objetivo é avaliar a capacidade de resposta do modelo, e não avaliar a qualidade do RAG
            retriever = vector_store.as_retriever(
                search_kwargs={"k": NUM_CHUNKS, "filter": {"md5": md5}}
            )

            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )

            try:
                result = invoke_com_retry(chain, row['question'])
            except RateLimitError as e:
                if 'insufficient_quota' in str(e):
                    logging.error('Créditos esgotados. Interrompendo avaliação RAG.')
                    break
                raise

            res_json = parse_llm_json(result['result'])
            res_docs = result['source_documents']

            registro = {
                'tipo': 'rag',
                'md5': md5,
                'question': row['question'],
                'data_category_QA': row['data_category_QA'],
                'pred': result['result'] if res_json is None else res_json['result'],
                'contexto': [doc.id for doc in res_docs],
                'model': model_name,
            }

            f.write(json.dumps(registro) + '\n')
            f.flush()
            # break

        # Avaliação Full Context (conteúdo completo do documento)
        pendentes_full = df[~df.apply(lambda r: (r['md5'], r['question'], 'full_context') in processados, axis=1)]
        logging.info(f'{len(pendentes_full)} questões pendentes para avaliação Full Context')

        for _, row in tqdm(pendentes_full.iterrows(), total=len(pendentes_full), desc='Full Context', unit='q'):
            md5 = row['md5']

            prompt_filled = QA_CHAIN_PROMPT.format(context=row['content'], question=row['question'])
            try:
                result = invoke_com_retry(llm, prompt_filled)
            except RateLimitError as e:
                if 'insufficient_quota' in str(e):
                    logging.error('Créditos esgotados. Interrompendo avaliação Full Context.')
                    break
                raise

            res_json = parse_llm_json(result.content)

            registro = {
                'tipo': 'full_context',
                'md5': md5,
                'question': row['question'],
                'data_category_QA': row['data_category_QA'],
                'pred': result.content if res_json is None else res_json['result'],
                'contexto': None,
                'model': model_name,
            }

            f.write(json.dumps(registro) + '\n')
            f.flush()


    df_output_file = pd.read_json(output_file, lines=True)
    metrics = show_metrics(df_output_file)
    registrar_experimento(model_name, metrics, n_chunks=NUM_CHUNKS)

    logging.info('Script concluído com sucesso!')
    

if __name__ == "__main__":
    main()