from fastapi import FastAPI
from pydantic import BaseModel
from langdetect import detect
import regex as re
from underthesea import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import unicodedata

app = FastAPI(title="AI NLP Service - Enhanced")

@app.on_event("startup")
async def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Stopword sets
en_stopwords = set(stopwords.words('english'))
vi_stopwords = {
    "và","là","của","có","cho","trong","một","những","với",
    "được","không","cũng","rất","thì","đã","này","đó","khi","ra","ở","như"
}

# Lemmatizer for English
wnl = WordNetLemmatizer()

class Query(BaseModel):
    text: str

def normalize_text(text: str) -> str:
    # remove diacritics, normalize spaces
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
    text = normalize_text(raw_text)
    lang = detect_language(text)

    # Tokenize
    if lang.startswith('vi'):
        tokens = word_tokenize(text, format='text').split()
        stopwords_set = vi_stopwords
        # Vietnamese lemmatization placeholder: keep tokens
        lemmas = tokens
    else:
        tokens = re.findall(r"\p{L}+", text.lower())
        stopwords_set = en_stopwords
        # English lemmatization
        lemmas = [wnl.lemmatize(tok) for tok in tokens]

    # Filter tokens and lemmas
    filtered_tokens = [t for t in tokens if t.isalpha() and t not in stopwords_set]
    filtered_lemmas = [l for l in lemmas if l.isalpha() and l not in stopwords_set]

    return {
        'lang': lang,
        'tokens': filtered_tokens,
        'lemmas': filtered_lemmas
    }
