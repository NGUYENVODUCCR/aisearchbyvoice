from fastapi import FastAPI
from pydantic import BaseModel
from langdetect import detect
import regex as re
from underthesea import word_tokenize
from nltk.corpus import stopwords
import nltk

app = FastAPI()

@app.on_event("startup")
async def download_nltk_data():
    nltk.download('stopwords', quiet=True)

STOPWORDS_EN = set(stopwords.words('english'))
STOPWORDS_VI = {
    "và","là","của","có","cho","trong","một","những","với",
    "được","không","cũng","rất","thì","đã","này","đó","khi","ra","ở","như"
}

class Query(BaseModel):
    text: str

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return 'unknown'

@app.post('/nlp/clean')
async def clean_text(q: Query):
    text = q.text.strip()
    lang = detect_language(text)
    
    if lang.startswith('vi'):
        raw_tokens = word_tokenize(text, format="text").split()
        stopwords_set = STOPWORDS_VI
    else:
        raw_tokens = re.findall(r"\p{L}+", text.lower())
        stopwords_set = STOPWORDS_EN
    
    filtered = [
        t for t in raw_tokens
        if t.isalpha() and t not in stopwords_set
    ]
    return {'lang': lang, 'tokens': filtered}
