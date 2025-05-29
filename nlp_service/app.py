from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langdetect import detect
import regex as re
from underthesea import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import unicodedata

app = FastAPI(title="AI NLP Service - Enhanced Tokens Vietnamese")

@app.on_event("startup")
async def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Stopword sets
en_stopwords = set(stopwords.words('english'))
vi_stopwords = {"và","là","của","có","cho","trong","một","những","với",
                "được","không","cũng","rất","thì","đã","này","đó","khi","ra","ở","như"}

# Lemmatizer for English
wnl = WordNetLemmatizer()

class Query(BaseModel):
    text: str


def normalize_text(text: str, is_vi: bool) -> str:
    # Normalize spaces; remove diacritics only for non-Vietnamese
    text = re.sub(r"\s+", " ", text).strip()
    if not is_vi:
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
    return text


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return 'unknown'


def generate_ngrams(tokens, max_n=3):
    ngrams = []
    length = len(tokens)
    for n in range(2, min(max_n, length) + 1):
        for i in range(length - n + 1):
            ngrams.append('_'.join(tokens[i:i+n]))
    return ngrams

@app.post('/nlp/clean')
async def clean_text(q: Query):
    raw_text = q.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Empty text provided.")

    # Detect language before normalizing
    lang = detect_language(raw_text)
    is_vi = lang.startswith('vi')

    # Normalize text conditionally
    text = normalize_text(raw_text, is_vi)

    # Tokenize & lemmatize
    if is_vi:
        tokens = word_tokenize(text, format='text').split()
        stop_set = vi_stopwords
        lemmas = tokens
    else:
        tokens = re.findall(r"\p{L}+", text.lower())
        stop_set = en_stopwords
        lemmas = [wnl.lemmatize(tok) for tok in tokens]

    # Filter stopwords
    filtered = [t for t in lemmas if t.isalpha() and t not in stop_set]

    # Generate bi- and tri-grams as combined tokens
    ngrams = generate_ngrams(filtered, max_n=3)
    enhanced_tokens = filtered + ngrams

    ngrams_display = [ng.replace('_', ' ') for ng in ngrams]
    return {
        'lang': lang,
        'tokens': enhanced_tokens,  # for backend matching
        'display_ngrams': ngrams_display  # for UI display without underscores
    }
