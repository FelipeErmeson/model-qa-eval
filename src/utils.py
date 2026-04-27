import hashlib
import tiktoken
import json
import re
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
tokenizer = tiktoken.get_encoding("cl100k_base") # encoding padrão GPT-4/3.5

def gerar_hash_md5(txt: str):
    txt_bytes = txt.encode('utf-8')
    hash_md5 = hashlib.md5(txt_bytes).hexdigest()
    return hash_md5

def length_function_tkt(text: str) -> int:
    return len(tokenizer.encode(text))

def carregar_dataset(filename='src/data/train-00000-of-00001.parquet'):
    colunas = ['content', 'question', 'data_category_QA']
    df = None

    try:    
        df = pd.read_parquet(path=filename, columns=colunas)
        print('Dataset lido com sucesso! shape:', df.shape)
    except Exception as ex:
        logging.error(f"Erro ao tentar ler dataset. {ex}")

    return df

def parse_llm_json(raw_text):
    # Procura o padrão de um objeto JSON { ... } ou lista [ ... ]
    # O re.DOTALL garante que o '.' capture quebras de linha
    match = re.search(r'(\{.*\}|\[.*\])', raw_text, re.DOTALL)
    
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar JSON: {e}")
            return None
            
    print("Nenhum JSON encontrado na string.")
    return None