import re

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_for_sending(text: str, max_chars: int = 20000) -> str:
    text = clean_text(text)
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[...TRUNCADO]"
    return text

def extract_keywords(text: str, top_k: int = 10):
    text = clean_text(text.lower())
    # remove números, pontuação e stopwords básicas
    words = re.sub(r"[^a-zA-ZÀ-ú ]", " ", text).split()
    stopwords = {"de","da","do","para","por","com","uma","um","em","e","o","a","que",
                 "na","no","nos","nas","pois","mas","ou","se","onde","como",
                 "tem","têm","ter","ser"}
    keywords = [w for w in words if w not in stopwords and len(w) > 3]
    # mantém ordem e remove duplicados
    seen = set()
    out = []
    for w in keywords:
        if w not in seen:
            out.append(w)
            seen.add(w)
        if len(out) >= top_k:
            break
    return out
