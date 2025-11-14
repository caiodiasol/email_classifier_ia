# llm_client.py
"""
Cliente LLM aprimorado para classificação e geração de resposta.

Funcionalidades:
- Carrega OPENAI_API_KEY via dotenv
- Anonimiza dados sensíveis minimamente (emails, CPFs/CNPJs, sequências longas de dígitos)
- Separa duas etapas: classify_email(...) e generate_response(...)
- Validação robusta do JSON retornado pelo modelo (evita falhas por "JSON quebrado")
- Heurística de fallback rápido se o LLM falhar
- Retry/backoff para chamadas à API
- call_llm_for_classify_and_respond(email_text) mantido por compatibilidade (retorna category, suggested_response)
"""

import os
import time
import re
import json
import openai
from typing import Tuple, Optional, Dict, Any
from dotenv import load_dotenv

# Carregar .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "20"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não configurada no ambiente")

openai.api_key = OPENAI_API_KEY

# Templates e configurações
INSTITUTIONAL_TONE = (
    "Use um tom institucional: cordial, profissional, objetivo e claro. "
    "Evite jargões técnicos quando não solicitados. Não invente prazos ou informações. "
    "Se precisar de dados adicionais para prosseguir, peça educadamente por 'dados de identificação' "
    "sem solicitar informações sensíveis (como CPF ou número de cartão) diretamente no e-mail."
)

CLASSIFY_PROMPT_INSTRUCTIONS = (
    "Classifique o e-mail em exatamente UMA das categorias: 'Produtivo' ou 'Improdutivo'.\n"
    "Forneça também um resumo curto (uma frase) e uma estimativa de confiança entre 0 e 1.\n"
    "Responda APENAS em JSON com as chaves: category, confidence, summary.\n"
    "category deve ser 'Produtivo' ou 'Improdutivo'. confidence deve ser um número entre 0.0 e 1.0.\n"
)

RESPONSE_PROMPT_INSTRUCTIONS = (
    "Com base na categoria, gere uma resposta curta (2-6 frases) adequada ao tom institucional.\n"
    "Se a categoria for 'Produtivo', sugira próximos passos claros e peça informações ausentes se necessário.\n"
    "Se 'Improdutivo', responda cordialmente sem abrir ticket.\n"
    "Não inclua instruções internas; responda como se fosse enviar ao cliente.\n"
    "Retorne apenas o texto da resposta (sem JSON)."
)

# Fallback templates (quando LLM falhar)
TEMPLATES = {
    "Produtivo": (
        "Olá,\n\nRecebemos sua solicitação e já estamos verificando. "
        "Por favor, confirme os dados necessários (número do processo ou mais detalhes) para que possamos acelerar o atendimento. "
        "Retornaremos com uma atualização assim que possível.\n\nAtenciosamente,"
    ),
    "Improdutivo": (
        "Olá,\n\nAgradecemos a sua mensagem. Caso precise de suporte ou queira abrir uma solicitação, por favor nos informe os detalhes."
    )
}

# Heurística simples para fallback
_PRODUCTIVE_KEYWORDS = [
    "erro", "problema", "solicit", "atualiza", "atualização", "status",
    "não consigo", "não funciona", "falha", "incidente", "reclama", "ajuda", "suporte"
]

# -------------------------
# Utilitários
# -------------------------
def _retry_backoff_call(func, retries=3, base_delay=1.0, *args, **kwargs):
    last_exc = None
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            delay = base_delay * (2 ** i)
            time.sleep(delay)
    raise last_exc

def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extrai o JSON mais à direita/primeiro encontrado no texto.
    Retorna dict ou lança ValueError.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Texto vazio ao extrair JSON")

    # procurar pelo primeiro '{' e o último '}' plausível
    first = text.find('{')
    last = text.rfind('}')
    if first == -1 or last == -1 or last < first:
        raise ValueError("Nenhum JSON encontrado no texto")
    json_text = text[first:last+1]

    # tentar corrigir vírgulas finais, aspas ' em vez de ", etc. (tenta uma correção simples)
    try:
        parsed = json.loads(json_text)
        return parsed
    except Exception:
        # tentativas simples de limpeza
        cleaned = json_text.replace("'", '"')
        cleaned = re.sub(r",\s*}", "}", cleaned)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        parsed = json.loads(cleaned)  # se falhar, propa
        return parsed

def anonymize_text(text: str) -> str:
    """
    Remoção/anonimização simples:
    - emails -> [EMAIL_REMOVIDO]
    - CPF/CNPJ (padrões comuns) -> [PII_REMOVIDO]
    - sequências longas de dígitos (6+) -> [NUM_REMOVIDO]
    Mantém legibilidade do texto para o LLM.
    """
    if not isinstance(text, str):
        return ""

    # emails
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL_REMOVIDO]", text)

    # cpf/cnpj (padrões simples)
    text = re.sub(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "[PII_REMOVIDO]", text)
    text = re.sub(r"\b\d{11}\b", "[PII_REMOVIDO]", text)
    text = re.sub(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b", "[PII_REMOVIDO]", text)
    text = re.sub(r"\b\d{14}\b", "[PII_REMOVIDO]", text)

    # sequências longas de dígitos (6 ou mais)
    text = re.sub(r"\b\d{6,}\b", "[NUM_REMOVIDO]", text)

    return text

def heuristics_fallback_classify(text: str) -> Tuple[str, float, str]:
    """
    Heurística simples: procura por palavras-chave
    Retorna: (category, confidence, summary)
    """
    t = (text or "").lower()
    for kw in _PRODUCTIVE_KEYWORDS:
        if kw in t:
            # confiança moderada
            return "Produtivo", 0.65, f"Contém palavra-chave indicativa: '{kw}'"
    # default
    return "Improdutivo", 0.55, "Nenhuma palavra-chave produtiva detectada"

# -------------------------
# Core: classificação
# -------------------------
def classify_email(email_text: str, max_tokens: int = 256, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Classifica o email e retorna um dict:
    { "category": "Produtivo"/"Improdutivo", "confidence": float, "summary": "resumo curto" }
    """
    text = anonymize_text(email_text)
    system = "Você é um assistente que classifica e resume e-mails para triagem em uma empresa financeira. Responda em Português."
    user = CLASSIFY_PROMPT_INSTRUCTIONS + "\nEmail: '''" + text + "'''\n\nSaída JSON:"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    def _call():
        return openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=TIMEOUT
        )

    try:
        resp = _retry_backoff_call(_call, retries=3, base_delay=1.0)
        text_out = resp["choices"][0]["message"]["content"].strip()
        parsed = _extract_json_from_text(text_out)

        # validação e normalização
        category = str(parsed.get("category", "")).strip()
        confidence = parsed.get("confidence", None)
        summary = parsed.get("summary", "") or parsed.get("summary", "")

        # normalizar categoria
        cat_norm = category.capitalize()
        if cat_norm not in ("Produtivo", "Improdutivo"):
            cat_norm = cat_norm.lower()
            if "prod" in cat_norm:
                cat_norm = "Produtivo"
            elif "improd" in cat_norm or "não" in cat_norm:
                cat_norm = "Improdutivo"
            else:
                # heurística se estiver incerto
                h_cat, h_conf, h_sum = heuristics_fallback_classify(email_text)
                return {"category": h_cat, "confidence": h_conf, "summary": h_sum}

        # garantir confidence numérico
        try:
            confidence = float(confidence)
            # clamp
            confidence = max(0.0, min(1.0, confidence))
        except Exception:
            confidence = 0.7 if cat_norm == "Produtivo" else 0.6

        summary = str(summary).strip() if summary else ""

        return {"category": cat_norm, "confidence": confidence, "summary": summary}

    except Exception as e:
        # fallback heurístico
        h_cat, h_conf, h_sum = heuristics_fallback_classify(email_text)
        return {"category": h_cat, "confidence": h_conf, "summary": h_sum}

# -------------------------
# Core: geração de resposta
# -------------------------
def generate_response(email_text: str, category: str, summary: Optional[str] = None,
                      max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Gera uma resposta apropriada ao e-mail com base na categoria.
    Retorna dict: {"suggested_response": str}
    """
    text = anonymize_text(email_text)
    # se improdutivo, usar template curto + LLM opcional para reescrever
    if category == "Improdutivo":
        # pedimos ao LLM para reescrever curtamente (para naturalidade), mas temos fallback
        prompt = (
            f"{INSTITUTIONAL_TONE}\n\n"
            f"O e-mail a seguir parece improdutivo. Gere uma resposta curta e cordial em Português.\n\n"
            f"Email: '''{text}'''\n\nResposta:"
        )

        messages = [
            {"role": "system", "content": "Assistente que gera respostas institucionais em Português."},
            {"role": "user", "content": prompt}
        ]

        def _call():
            return openai.ChatCompletion.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens,
                request_timeout=TIMEOUT
            )

        try:
            resp = _retry_backoff_call(_call, retries=2, base_delay=0.8)
            reply = resp["choices"][0]["message"]["content"].strip()
            # segurança: não retornar textos muito longos
            if len(reply) < 30:
                raise ValueError("Resposta curta demais")
            return {"suggested_response": reply}
        except Exception:
            return {"suggested_response": TEMPLATES["Improdutivo"]}

    # PRODUTIVO
    # montar prompt que inclua summary se houver
    prompt = (
        f"{INSTITUTIONAL_TONE}\n\n"
        f"O e-mail abaixo foi classificado como 'Produtivo'. Gere uma resposta profissional em Português, "
        f"objetiva, com próximos passos claros e solicite informações adicionais se necessário.\n\n"
        f"Resumo do e-mail: \"{summary or ''}\"\n\n"
        f"Email: '''{text}'''\n\nResposta:"
    )

    messages = [
        {"role": "system", "content": "Assistente que gera respostas institucionais e seguras em Português."},
        {"role": "user", "content": prompt}
    ]

    def _call_prod():
        return openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
            request_timeout=TIMEOUT
        )

    try:
        resp = _retry_backoff_call(_call_prod, retries=3, base_delay=1.0)
        reply = resp["choices"][0]["message"]["content"].strip()
        if not reply:
            raise ValueError("Resposta vazia do LLM")
        return {"suggested_response": reply}
    except Exception:
        return {"suggested_response": TEMPLATES["Produtivo"]}

# -------------------------
# Função compatibilidade antiga (interface simples)
# -------------------------
def call_llm_for_classify_and_respond(email_text: str) -> Tuple[str, str]:
    """
    Função compatível com versões anteriores.
    Retorna (category, suggested_response).
    Internamente usa classify_email(...) e generate_response(...).
    """
    # classificar
    cls = classify_email(email_text)
    category = cls.get("category", "Improdutivo")
    summary = cls.get("summary", "")

    # gerar resposta
    gen = generate_response(email_text, category, summary=summary)
    suggested = gen.get("suggested_response", TEMPLATES.get(category, ""))

    return category, suggested

# -------------------------
# Exportar utilitários (opcionais)
# -------------------------
__all__ = [
    "classify_email",
    "generate_response",
    "call_llm_for_classify_and_respond",
    "anonymize_text"
]
