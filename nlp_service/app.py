from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langdetect import detect
import regex as re
from underthesea import word_tokenize
from nltk.corpus import stopwords
import nltk
import spacy
import whisper
import tempfile
from typing import Union

app = FastAPI(title="AI NLP Service")

# --- Startup: download resources & initialize models ---
@app.on_event("startup")
async def setup_nlp_models():
    # NLTK stopwords
    nltk.download('stopwords', quiet=True)
    global STOPWORDS_EN, STOPWORDS_VI
    STOPWORDS_EN = set(stopwords.words('english'))
    STOPWORDS_VI = {
        "và","là","của","có","cho","trong","một","những","với",
        "được","không","cũng","rất","thì","đã","này","đó","khi","ra","ở","như"
    }

    # SpaCy models for advanced NLP
    try:
        nlp_en = spacy.load("en_core_web_sm")
    except:
        spacy.cli.download("en_core_web_sm")
        nlp_en = spacy.load("en_core_web_sm")
    try:
        nlp_vi = spacy.load("vi_spacy")
    except:
        nlp_vi = None

    # Whisper for optional ASR (if audio input)
    asr_model = whisper.load_model("base")

    app.state.nlp = { 'en': nlp_en, 'vi': nlp_vi }
    app.state.asr = asr_model

# Pydantic model for text input
type TextQuery = BaseModel(text: str)

@app.post('/nlp/clean')
async def clean_text(
    query: Union[TextQuery, None] = None,
    audio_file: UploadFile = File(None)
):
    # 1. Obtain text: from JSON or transcribe audio
    if audio_file:
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=415, detail="Upload audio file only.")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        content = await audio_file.read()
        tmp.write(content); tmp.flush()
        text = app.state.asr.transcribe(tmp.name)['text'].strip()
    elif query:
        text = query.text.strip()
    else:
        raise HTTPException(status_code=400, detail="Provide 'text' JSON or 'audio_file'.")

    # 2. Language detection
    try:
        lang = detect(text)
    except:
        lang = 'unknown'

    # 3. Tokenization & stopword removal
    if lang.startswith('vi'):
        raw = word_tokenize(text, format='text').split()
        stop_set = STOPWORDS_VI
        nlp = app.state.nlp['vi']
    else:
        raw = re.findall(r"\p{L}+", text.lower())
        stop_set = STOPWORDS_EN
        nlp = app.state.nlp['en']
    tokens = [t for t in raw if t.isalpha() and t not in stop_set]

    # 4. Advanced NLP: lemmas, POS, entities
    if nlp:
        doc = nlp(" ".join(tokens))
        lemmas = [tok.lemma_ for tok in doc]
        pos_tags = [(tok.text, tok.pos_) for tok in doc]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
    else:
        lemmas, pos_tags, entities = tokens, [], []

    # 5. Return structured analysis
    return JSONResponse({
        'lang': lang,
        'tokens': tokens,
        'lemmas': lemmas,
        'pos_tags': pos_tags,
        'entities': entities
    })
