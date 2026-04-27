

template = """Você é um profissional especializado em Perguntas e Respostas.
Seu OBJETIVO é verificar se é possível responder a pergunta APENAS com os dados contidos no contexto.

Responda sempre em formato JSON válido. NÃO INSIRA nenhuma explicação ou conteúdo qualquer antes ou depois do json.
Gere APENAS UM json.

Caso as informações do contexto NÃO SEJAM suficientes, incompletas ou não estejam presentes no contexto para responder a pergunta, responda negativo.
Exemplo de formato esperado:
{{
    "result": "negativo"
}}

Caso as informações do contexto ESTEJAM PRESENTES e SEJAM suficientes para responder corretamente a pergunta, responda positivo.
Exemplo de formato esperado:
{{
    "result": "positivo"
}}

Se você não souber a resposta, não tente inventar.

Contexto: {context}

Pergunta: {question}

JSON de Saída:"""

template_com_justificativa = """Você é um profissional especializado em Perguntas e Respostas.
Seu OBJETIVO é verificar se é possível responder a pergunta APENAS com os dados contidos no contexto.

Responda sempre em formato JSON válido. NÃO INSIRA nenhuma explicação ou conteúdo qualquer antes ou depois do json.
Gere APENAS UM json.

Caso as informações do contexto NÃO SEJAM suficientes, incompletas ou não estejam presentes no contexto para responder a pergunta, responda negativo.
Exemplo de formato esperado:
{{
    "justificativa": "O contexto não contém informações sobre X.",
    "result": "negativo"
}}

Caso as informações do contexto ESTEJAM PRESENTES e SEJAM suficientes para responder corretamente a pergunta, responda positivo.
Exemplo de formato esperado:
{{
    "justificativa": "O contexto menciona explicitamente que X é Y, o que responde diretamente a pergunta.",
    "result": "positivo"
}}

Se você não souber a resposta, não tente inventar.

Contexto: {context}

Pergunta: {question}

JSON de Saída:"""