# Email Classifier IA

Aplicação full stack que classifica e-mails e sugere respostas automáticas usando modelos da OpenAI. O frontend (HTML/Tailwind/JS) roda na Vercel, enquanto o backend Flask roda na Render.

## Stack

- Frontend: HTML + Tailwind (CDN) + JavaScript vanilla
- Backend: Python 3.11+, Flask, pdfminer.six, OpenAI API, Gunicorn
- Infra: Vercel (frontend), Render (backend)

## Pré-requisitos

- Python 3.11+
- Conta e chave da OpenAI (`OPENAI_API_KEY`)
- Opcional: Node ou servidor estático para servir o frontend durante o desenvolvimento (pode abrir direto no navegador)

## Configuração local

### 1. Clonar o repositório

```bash
git clone https://github.com/caiodiasol/email_classifier_ia.git
cd email_classifier_ia
```

### 2. Backend

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

Crie `backend/.env`:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini      # opcional
OPENAI_TIMEOUT=20             # opcional
MAX_EMAIL_CHARS=12000         # opcional
```

Execute:

```bash
cd backend
flask run --port 5000         # ou python app.py / gunicorn app:app
```

### 3. Frontend

Abra `frontend/index.html` direto no navegador ou sirva via `live-server`. Para apontar ao backend local, garanta:

```javascript
const API_BASE_URL = "http://localhost:5000";
```

### 4. Teste integrado

1. Backend rodando em `http://localhost:5000`.
2. Frontend carregado localmente.
3. Envie texto/arquivo e verifique o retorno JSON.

## Deploy

### Render (backend)

1. Root Directory: `backend/`
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `gunicorn app:app`
4. Variáveis de ambiente: `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_TIMEOUT`, `MAX_EMAIL_CHARS`
5. CORS configurado em `backend/app.py` para liberar o domínio do frontend.

### Vercel (frontend)

1. Importar o repo via GitHub.
2. Root Directory: `frontend/`
3. Build Command: _(vazio)_
4. Output Directory: `.`
5. Certifique-se de que `API_BASE_URL` aponta para a URL pública do backend (ex.: `https://email-classifier-ia-8iqe.onrender.com`).

## Endpoint principal

`POST /api/process`

Form-data:

- `text` (opcional)
- `file` (PDF ou TXT)

Resposta bem-sucedida:

```json
{
  "category": "Produtivo",
  "suggested_response": "...",
  "confidence": 0.87,
  "summary": "Resumo curto"
}
```

## Troubleshooting

- **CORS error**: cheque a lista de origens em `backend/app.py`.
- **OpenAI indisponível**: o backend usa heurística fallback. Verifique logs no Render.
- **Arquivos ignorados**: confirme `.gitignore` para garantir que `backend/requirements.txt` e `.env.example` (se houver) estejam versionados.

## Scripts úteis

```bash
pip install -r backend/requirements.txt
cd backend && gunicorn app:app
```

## Licença

MIT License. Veja `LICENSE`.

