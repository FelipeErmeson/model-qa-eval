# Model Q&A Eval

## 1. Apresentação

Este é um projeto para avaliação de respostas de modelos generativos Q&A.

## 2. Instruções para rodar o projeto

### Pré-requisitos

- Python 3.10+
- pip ou conda
- Criar conta (free tier) na Pinecone e gerar a chave de API
- Criar conta na OpenAI e gerar a chave de API

### 1. Clone o repositório
```bash
git clone url-de-clone
cd model-qa-eval
```

### 2. Instale as dependências
Recomenda-se o uso de um ambiente virtual:
```bash
python -m venv model-qa-eval
# Ative o ambiente virtual:
# No Windows:
model-qa-eval\Scripts\activate
# No Linux/Mac:
source model-qa-eval/bin/activate
pip install -r requirements.txt
```
Para executar algum notebook, execute:
```bash
pip install -e .
```

### 3. Configuração de API Keys
Crie na raíz do seu projeto um arquivo chamado **.env**

Nele insira as duas chaves de APIs criadas:
```
PINECONE_API_KEY = "....."
OPENAI_API_KEY = "....."
```

### 4. Ingestão do dataset no Pinecone
Esta seção irá ser responsável por inserir o dataset em um banco de dados vetorial (Pinecone).
O script irá realizar a divisão dos documentos em chunks e gerar os respectivos embeddings.
Na pasta raíz e com o ambiente virtual ativado, execute:
```bash
python -m src.ingestao.insert_dataset_pinecone nome-do-seu-index
```
Caso ocorra algum problema inesperado, veja no console do terminal em qual batch parou, e execute a partir dele. Exemplo:
```bash
python -m src.ingestao.insert_dataset_pinecone nome-do-seu-index --start-batch 45
```

### 5. Execução do pipeline de avaliação do modelo
Este script irá rodar o modelo escolhido para todo o dataset.
Note que são realizados dois tipos de avaliação:

- 1. RAG: O modelo é avaliado apenas com os chunks mais semelhantes.
- 2. Full Context: O modelo é avaliado com TODO o conteúdo do documento.

Na pasta raíz, execute:
```bash
python -m src.main nome-do-seu-index --model-name gpt-4o-mini
```

Note que se você não passar nenhum modelo, o modelo padrão a ser executado será: gpt-5.4-nano

Perceba que o script só aceitará modelos do tipo Chat Completions e da OpenAI.

### 6. Notebooks

* avaliacao_resultados.ipynb - Notebook responsável por avaliar resultados do modelo gpt-5.4-nano. Quando se executa o pipeline principal (main.py), já é gerado um arquivo resultados.jsonl. Porém, caso precise do csv com o respectivo id da tabela principal. Rode este notebook.

* estimativa_custo.ipynb - Notebook auxiliar responsável por estimar custo ao rodar o pipeline principal de um determinado modelo.


## 3. Estrutura de Arquivos

```
model-qa-eval/
├── README.md                                         (este arquivo)
├── pyproject.toml                                    (configuração do pacote)
├── requirements.txt                                  (arquivo de instalação de pacotes Python)
├── .env                                              (seu arquivo de tokens de APIs)
├── experiments.csv                                   (arquivo responsável por registrar os experimentos a cada run)
├── .vscode/
    └── launch.json                                   (configurações de debug)
├── src/
│   ├── main.py                                       (pipeline de avaliação do modelo)
│   ├── error_analysis.py                             (script para análise de erros)
│   ├── utils.py                                      (funções auxiliares)
│   ├── data/
│   │   ├── train-00000-of-00001.parquet              (dataset original)
│   │   └── {model_name}/
│   │       └── resultados.jsonl                      (resultados por modelo avaliado)
│   │       └── results.csv                           (resultados por modelo avaliado com id do dataset original)
│   ├── ingestao/
│   │   └── insert_dataset_pinecone.py                (ingestão dos chunks no Pinecone)
│   ├── services/
│   │   ├── auth_service.py                           (autenticação Pinecone)
│   │   └── index_service.py                          (gerenciamento de indexes)
│   ├── prompts/
│   │   └── prompts_template.py                       (template do prompt de avaliação)
│   └── notebooks/
│       ├── estimativa_custo.ipynb                    (estimativa de tokens e custo)
│       └── avaliacao_resultados.ipynb                (gera métricas, análise dos resultados e salva arquivo csv)

```

## 4. Características e Decisões Técnicas

- **LangChain**: Um framework de código aberto poderoso para criação de agentes e soluções RAG.
- **Pinecone**: Banco de dados vetorial escalável e ótima integração com o langchain. Além da versão free tier possuir capacidade de inserir todo os dados disponibilizados e não ter nenhum custo.
- **Modelos OpenAI**: Escolhi modelos como gpt-5.4-nano e gpt-4o-mini para os experimentos, exclusivamente pelo custo econômico.
- **Modelo de embeddings**: O modelo escolhido para realizar os embeddings dos chunks foi o text-embedding-3-small, por motivos de custo.

## 5. Resultados

### 1. Experimento com o modelo gpt-5.4-nano

**Resultado utilizando chunks mais relevantes:**

| classe | precision | recall | f1-score | support |
|---|---|---|---|---|
| negativo | 0.70 | 0.97 | 0.81 | 1566 |
| positivo | 0.96 | 0.62 | 0.75 | 1720 |
| accuracy | | | 0.79 | 3286 |
| macro avg | 0.83 | 0.79 | 0.78 | 3286 |
| weighted avg | 0.83 | 0.79 | 0.78 | 3286 |

**Resultado utilizando todo o documento (full context):**

| classe | precision | recall | f1-score | support |
|---|---|---|---|---|
| negativo | 0.75 | 0.96 | 0.84 | 1566 |
| positivo | 0.95 | 0.71 | 0.82 | 1720 |
| accuracy | | | 0.83 | 3286 |
| macro avg | 0.85 | 0.84 | 0.83 | 3286 |
| weighted avg | 0.86 | 0.83 | 0.83 | 3286 |

  **Por que o RAG tem recall menor nos positivos?**

  Para um caso positivo, a resposta existe no documento. O problema é que o RAG recupera apenas 3 chunks de 512 tokens — e mesmo com o filtro por md5, a seleção é por similaridade semântica. Se o chunk
  que contém a resposta não estiver entre os top-3 recuperados, o modelo vê um contexto incompleto e tende a classificar como negativo (não encontrei a resposta → é negativo).

  Isso gera falsos negativos para a classe positivo → recall baixo.

  Para os negativos, a resposta não existe em nenhum chunk do documento — qualquer 3 chunks que o retriever retornar ainda mostram que a resposta não está lá → acerto quase certo → recall altíssimo
  (97%).

  O full context resolve isso porque o modelo sempre recebe o documento inteiro, nunca perde a informação relevante.

  Resumindo:

  O RAG introduz um ruído de recuperação que penaliza os positivos. Isso é um problema clássico de RAG: a qualidade da resposta depende da qualidade da recuperação. O objetivo aqui é avaliar o modelo, não o RAG — o que justifica plenamente o full context ter resultado superior nesse experimento.


### 2. Experimento com o modelo gpt-4o-mini

**Resultado utilizando chunks mais relevantes:**

| classe | precision | recall | f1-score | support |
|---|---|---|---|---|
| negativo | 0.63 | 0.98 | 0.77 | 1566 |
| positivo | 0.96 | 0.48 | 0.64 | 1720 |
| accuracy | | | 0.72 | 3286 |
| macro avg | 0.80 | 0.73 | 0.71 | 3286 |
| weighted avg | 0.81 | 0.72 | 0.70 | 3286 |

**Resultado utilizando todo o documento (full context):**

| classe | precision | recall | f1-score | support |
|---|---|---|---|---|
| negativo | 0.68 | 0.98 | 0.80 | 1566 |
| positivo | 0.97 | 0.59 | 0.73 | 1720 |
| accuracy | | | 0.77 | 3286 |
| macro avg | 0.82 | 0.78 | 0.77 | 3286 |
| weighted avg | 0.83 | 0.77 | 0.77 | 3286 |


## 6. Conclusões Gerais

**Modelo adotado: `gpt-5.4-nano` com Full Context**

O `gpt-5.4-nano` supera o `gpt-4o-mini` em todas as métricas relevantes — surpreendente considerando que o `gpt-4o-mini` é geralmente considerado mais capaz. Isso sugere que o `gpt-5.4-nano` segue melhor as instruções do prompt para esta tarefa específica de classificação binária.

Ambos os modelos demonstram capacidade para a tarefa, com a ressalva de um padrão claro de comportamento: são **conservadores** — quando há dúvida, tendem a classificar como *negativo* (não encontrei a resposta). Isso explica o recall baixo nos positivos e altíssimo nos negativos em todos os cenários.

Esse comportamento é esperado e até desejável em aplicações reais de Q&A, onde um falso positivo (inventar uma resposta) é mais prejudicial do que um falso negativo (admitir que não sabe).

O **Full Context é superior** ao RAG para esta tarefa por uma razão estrutural: o objetivo é avaliar o modelo, não o retriever — e o RAG introduz uma variável de ruído (qualidade da recuperação) que penaliza artificialmente o recall nos positivos.

## 7. Análise de Erros

É possível executar o script para análise mais detalhada dos erros.

```bash
python -m src.error_analysis gpt-5.4-nano --tipo full_context
```

Se tratando do modelo escolhido **gpt-5.4-nano**, 556 erros de 3286 (17%).
A grande maioria são FN (493) — o modelo errou muito mais "por omissão" (não encontrou a resposta) do que "por invenção" (FP: 63).

  Padrões nos Falsos Positivos

  1. Perguntas quantitativas — confirmado como padrão
  ▎ "Quantos operadores...", "Quantas unidades...", "Quantos salões...", "Quantas lojas..."

  2. Prazo/tempo de entrega — aparece 3 vezes
  ▎ "tempo de entrga do produto?", "Qual é o prazo de entrega...", "Qual é o tempo máximo de entrega..."
  ▎ O modelo parece confundir porque documentos de FAQ geralmente têm prazo de entrega — então ele presume que a resposta existe.

  3. Perguntas plausíveis dentro do domínio do documento
  ▎ "Como chamar suporte?", "Qual é o endereço de e-mail do atendimento?", "Qual é a validade do cartão oferta?"

  Essas são perguntas que deveriam estar num FAQ de e-commerce — o modelo alucina porque conhece o padrão do domínio pelo treinamento.

  4. Perguntas de conhecimento geral que não dependem do documento
  ▎ "Qual cidade abriga a Mesquita Azul?" — o modelo sabe a resposta pelo treinamento e diz que está no documento.

  ---
  **Diagnóstico central**

  O modelo erra nos FP não porque o documento é ambíguo, mas porque a pergunta é plausível dentro do domínio — ele infere que a resposta deveria estar lá e classifica como positivo. Isso é alucinação
  contextual, não confusão de leitura.

## 8. Limitações e Trabalhos Futuros

* Pipeline capaz de rodar apenas modelos OpenAI do tipo Chat Completions. O ideal seria receber apenas o nome do modelo e o provider, independente.

* Usar um LLM Judge pedindo ao modelo para justificar a resposta e o judge avaliasse a qualidade da justificativa.

* Realizar uma análise de consistência (temperatura). Rodar a mesma questão N vezes e medir a variância das respostas — revela se o modelo é confiável ou instável na tarefa.

* Melhorar a qualidade de recuperação de chunks, aumentando o número de chunks recuperados e depois uma aplicação de rerank.