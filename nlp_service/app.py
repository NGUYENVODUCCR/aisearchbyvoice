from fastapi import FastAPI
from pydantic import BaseModel
import re
from nltk.corpus import stopwords

app = FastAPI()
# Load stopwords tiếng Anh
STOPWORDS = set(stopwords.words('english'))
# Nếu muốn, mở rộng với stopwords tiếng Việt thủ công:
VIETNAMESE_STOPWORDS = {"và", "là", "của", "có", "cho", "trong", "một", "những", "với", "được", "không", "cũng", "rất", "thì", "đã", "này", "đó", "khi", "ra", "ở", "như"}
STOPWORDS.update(VIETNAMESE_STOPWORDS)

class Query(BaseModel):
    text: str

@app.post('/nlp/clean')
async def clean_text(q: Query):
    # Dùng regex simple để tokenization, tránh phụ thuộc punkt
    words = re.findall(r"\b\w+\b", q.text.lower())
    filtered = [t for t in words if t.isalpha() and t not in STOPWORDS]
    return {'tokens': filtered}