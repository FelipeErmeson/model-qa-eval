import argparse
import pandas as pd
from langdetect import detect, LangDetectException

from .utils import carregar_dataset, gerar_hash_md5, length_function_tkt


NEGATION_WORDS = [
    'não', 'nao', 'nunca', 'jamais', 'nem ',        # português
    'not', 'never', "don't", "doesn't", "isn't",    # inglês
    'no ', 'nunca', 'jamás', 'ni ',                  # espanhol
]


def detectar_idioma(text: str) -> str:
    try:
        return detect(str(text)[:500])
    except LangDetectException:
        return 'unknown'


def tem_negacao(text: str) -> bool:
    text_lower = str(text).lower()
    return any(word in text_lower for word in NEGATION_WORDS)


def classificar_erro(row) -> str:
    true = row['data_category_QA']
    pred = row['pred']
    if true == pred:
        return 'correct'
    if true == 'positivo' and pred == 'negativo':
        return 'FN'  # modelo disse "não tem resposta" mas tinha
    if true == 'negativo' and pred == 'positivo':
        return 'FP'  # modelo disse "tem resposta" mas não tinha
    return 'other'


def carregar_dados(model_name: str, tipo: str) -> pd.DataFrame:
    output_file = f'src/data/{model_name}/resultados.jsonl'

    df_pred = pd.read_json(output_file, lines=True)
    df_pred = df_pred[df_pred['tipo'] == tipo].drop_duplicates(subset=['md5', 'question'])

    df_true = carregar_dataset()
    df_true = df_true[df_true['data_category_QA'].isin(['positivo', 'negativo'])]
    df_true['md5'] = df_true['content'].apply(gerar_hash_md5)

    df = df_pred.merge(
        df_true[['md5', 'question', 'content', 'data_category_QA']].drop_duplicates(subset=['md5', 'question']),
        on=['md5', 'question'],
        how='left'
    )
    return df


def secao(titulo: str):
    print(f'\n{"=" * 60}')
    print(f'  {titulo}')
    print(f'{"=" * 60}')


def analisar(df: pd.DataFrame, model_name: str, tipo: str):
    df = df.copy()
    df['erro'] = df.apply(classificar_erro, axis=1)
    df['tokens_content'] = df['content'].apply(length_function_tkt)
    df['tokens_question'] = df['question'].apply(length_function_tkt)
    df['lang_question'] = df['question'].apply(detectar_idioma)
    df['tem_negacao'] = df['question'].apply(tem_negacao)

    erros = df[df['erro'].isin(['FN', 'FP'])]
    corretos = df[df['erro'] == 'correct']

    print(f'\nModelo: {model_name} | Tipo: {tipo}')
    print(f'Total: {len(df)} | Corretos: {len(corretos)} | Erros: {len(erros)}')
    print(f'FN (positivo classificado como negativo): {len(df[df["erro"] == "FN"])}')
    print(f'FP (negativo classificado como positivo): {len(df[df["erro"] == "FP"])}')

    # --- Comprimento do conteúdo ---
    secao('1. COMPRIMENTO DO CONTEÚDO (tokens)')
    print(df.groupby('erro')['tokens_content'].describe()[['mean', '50%', 'max']].round(0))

    # --- Idioma das perguntas ---
    secao('2. IDIOMA DAS PERGUNTAS')
    lang_total = df.groupby('lang_question')['erro'].count().rename('total')
    lang_erros = erros.groupby('lang_question')['erro'].count().rename('erros')
    lang_df = pd.concat([lang_total, lang_erros], axis=1).fillna(0)
    lang_df['taxa_erro (%)'] = (lang_df['erros'] / lang_df['total'] * 100).round(1)
    print(lang_df.sort_values('taxa_erro (%)', ascending=False))

    # --- Negação nas perguntas ---
    secao('3. NEGAÇÃO NA PERGUNTA')
    neg_df = df.groupby(['tem_negacao', 'erro']).size().unstack(fill_value=0)
    print(neg_df)

    fn = df[df['erro'] == 'FN']
    fp = df[df['erro'] == 'FP']
    taxa_neg_fn = fn['tem_negacao'].mean() * 100
    taxa_neg_fp = fp['tem_negacao'].mean() * 100
    taxa_neg_correct = corretos['tem_negacao'].mean() * 100
    print(f'\n% com negação nos FN:      {taxa_neg_fn:.1f}%')
    print(f'% com negação nos FP:      {taxa_neg_fp:.1f}%')
    print(f'% com negação nos corretos: {taxa_neg_correct:.1f}%')

    # --- Exemplos de erros ---
    secao('4. EXEMPLOS DE ERROS')

    print('\n--- 5 Falsos Negativos (tinha resposta, modelo disse não) ---')
    cols = ['question', 'data_category_QA', 'pred', 'tokens_content', 'lang_question']
    print(fn[cols].head(5).to_string(index=False))

    print('\n--- 5 Falsos Positivos (não tinha resposta, modelo disse que tinha) ---')
    print(fp[cols].head(5).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Nome do modelo avaliado')
    parser.add_argument('--tipo', type=str, default='full_context', choices=['rag', 'full_context'],
                        help='Tipo de avaliação (default: full_context)')
    args = parser.parse_args()

    df = carregar_dados(args.model_name, args.tipo)
    analisar(df, args.model_name, args.tipo)


if __name__ == '__main__':
    main()
