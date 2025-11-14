import os
from dotenv import load_dotenv

# Carregar variáveis ANTES de módulos dependentes
load_dotenv()

import tempfile
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pdfminer.high_level import extract_text
from nlp_utils import preprocess_for_sending, extract_keywords
from llm_client import (
    classify_email,
    generate_response,
    call_llm_for_classify_and_respond  # compatibilidade
)
from templates import TEMPLATES


app = Flask(__name__, static_folder='../frontend', static_url_path='/')

ALLOWED_EXTENSIONS = {'pdf', 'txt'}
MAX_SEND_CHARS = int(os.getenv("MAX_EMAIL_CHARS", "12000"))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(tmp_path, filename):
    if filename.lower().endswith('.pdf'):
        return extract_text(tmp_path) or ""
    else:
        try:
            with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return ""


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/process', methods=['POST'])
def process():
    text = (request.form.get('text') or '').strip()
    file = request.files.get('file')

    if not text and not file:
        return jsonify({"error": "Nenhum texto ou arquivo enviado"}), 400

    # Caso venha arquivo
    file_text = ""
    if file:
        filename = secure_filename(file.filename or "upload")
        if not allowed_file(filename):
            return jsonify({"error": "Tipo de arquivo não permitido"}), 400

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            file_text = extract_text_from_file(tmp.name, filename)

    # Unir texto + arquivo
    full_text = (text + "\n" + file_text).strip()
    if not full_text:
        return jsonify({"error": "Conteúdo do e-mail vazio após extração"}), 400

    # Preprocessamento com corte seguro
    to_send = preprocess_for_sending(full_text, max_chars=MAX_SEND_CHARS)

    # ============================================================
    # NOVO FLUXO — Classificação + Geração independente
    # ============================================================
    try:
        cls = classify_email(to_send)
        category = cls.get("category", "Improdutivo")
        confidence = cls.get("confidence", None)
        summary = cls.get("summary", "")

        # Normalização segura
        if category.capitalize() not in ("Produtivo", "Improdutivo"):
            category = "Produtivo" if any(k in to_send.lower()
                    for k in ["solic", "erro", "problema", "atualiza"]) else "Improdutivo"

        # gerar resposta
        gen = generate_response(to_send, category, summary=summary)
        suggested = gen.get("suggested_response", TEMPLATES.get(category, ""))

    except Exception:
        # Fallback — sem LLM
        fallback_cat = (
            "Produtivo" if any(k in full_text.lower()
                               for k in ["erro", "problema", "solicit", "preciso", "atualiza"])
            else "Improdutivo"
        )
        return jsonify({
            "category": fallback_cat,
            "suggested_response": TEMPLATES[fallback_cat],
            "warning": "Resposta por fallback; LLM indisponível."
        }), 200

    # Enriquecer resposta com keywords se houver placeholder
    keywords = extract_keywords(full_text, top_k=6)
    if '{keywords}' in suggested:
        suggested = suggested.replace('{keywords}', ', '.join(keywords))

    # ============================================================
    # RETORNO — agora inclui summary e confidence também
    # ============================================================
    return jsonify({
        "category": category,
        "suggested_response": suggested,
        "confidence": confidence,
        "summary": summary
    }), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)
