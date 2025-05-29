from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langdetect import detect
import regex as re
from underthesea import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import unicodedata

app = FastAPI(title="AI NLP Service - Text-Only")

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


def normalize_text(text: str) -> str:
    # Normalize unicode, remove diacritics and extra spaces
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return 'unknown'

@app.post('/nlp/clean')
async def clean_text(q: Query):
    raw_text = q.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Empty text provided.")

    # Normalize and detect language
    text = normalize_text(raw_text)
    lang = detect_language(text)

    # Tokenization & lemmatization
    if lang.startswith('vi'):
        tokens = word_tokenize(text, format='text').split()
        stop_set = vi_stopwords
        lemmas = tokens  # no lemmatizer for Vietnamese
    else:
        tokens = re.findall(r"\p{L}+", text.lower())
        stop_set = en_stopwords
        lemmas = [wnl.lemmatize(tok) for tok in tokens]

    # Filter tokens and lemmas
    filtered_tokens = [t for t in tokens if t.isalpha() and t not in stop_set]
    filtered_lemmas = [l for l in lemmas if l.isalpha() and l not in stop_set]

    return {
        'lang': lang,
        'tokens': filtered_tokens,
        'lemmas': filtered_lemmas
    }
