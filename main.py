from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timedelta
import re
from zoneinfo import ZoneInfo
from typing import Optional
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from vncorenlp import VnCoreNLP
from tzlocal import get_localzone

app = FastAPI()

# Load NER model and tokenizer
MODEL_PATH = "./model"  # Path to your saved model directory
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)  # Slow tokenizer for phoBERT
model.eval()

# Initialize VnCoreNLP for word segmentation
rdrsegmenter = VnCoreNLP(
    "./VnCoreNLP-1.1.1.jar",
    annotators="wseg",
    max_heap_size='-Xmx2g'
)

# Initialize NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Từ điển ánh xạ
TIME_DICT = {
    "sáng": {"hour": 8, "minute": 0},
    "chiều": {"hour": 14, "minute": 0},
    "tối": {"hour": 18, "minute": 0},
    "ngày_mai": {"days": 1},
    "mai": {"days": 1},
    "mốt": {"days": 2},
}
EVENT_DICT = {
    "họp": "họp",
    "đi chơi": "đi chơi"
}
DURATION_DICT = {
    "p": "minute",
    "phút": "minute",
    "h": "hour",
    "giờ": "hour",
    "tiếng": "hour",
    "ngày": "day"
}

# Additional event-related keywords for intent detection
INTENT_KEYWORDS = set(EVENT_DICT.keys()) | {"nhớ", "nhắc", "hẹn", "mai", "mốt"}

# Input model
class TextInput(BaseModel):
    text: str

# Intent detection based on keywords
def detect_intent(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in INTENT_KEYWORDS)

# NER processing with VnCoreNLP and pipeline
def predict_ner(text):
    # Segment the input text
    segmented = rdrsegmenter.tokenize(text)
    segmented_sentence = " ".join(segmented[0])  # Assume single sentence
    
    # Run NER pipeline
    results = ner_pipeline(segmented_sentence)
    
    # Format output to match merge_ner_tokens expectations
    ner_output = [
        {
            "entity": result["entity"],
            "word": result["word"],
            "score": float(result["score"])  # Convert np.float32 to float
        }
        for result in results
    ]
    
    return ner_output

# Ghép token bị tách
def merge_ner_tokens(ner_output):
    entities = {"TIME": [], "EVENT": [], "DURATION": []}
    current_entity = None
    current_tokens = []

    for token in ner_output:
        entity_type = token['entity'].split('-')[-1]  # TIME, EVENT, DURATION
        word = token['word']

        if token['entity'].startswith('B-'):
            if current_tokens:
                entities[current_entity].append(" ".join(current_tokens))
            current_entity = entity_type
            current_tokens = [word]
        elif token['entity'].startswith('I-') and current_entity == entity_type:
            current_tokens.append(word)

    if current_tokens and current_entity:
        entities[current_entity].append(" ".join(current_tokens))

    if entities["DURATION"]:
        entities["DURATION"] = ["".join(entities["DURATION"]).replace(" ", "")]

    return entities

# Chuẩn hóa TIME
def standardize_time(entities, timezone):
    current_datetime = datetime.now(tz=timezone)
    if not entities["TIME"]:
        dt = current_datetime + timedelta(hours=1)
        return dt.isoformat()

    time_str = " ".join(entities["TIME"]).lower()
    dt = current_datetime

    if "mai" in time_str or "ngày_mai" in time_str:
        dt = dt + timedelta(days=1)
    elif "mốt" in time_str:
        dt = dt + timedelta(days=2)

    if "sáng" in time_str:
        dt = dt.replace(hour=8, minute=0)
    elif "chiều" in time_str:
        dt = dt.replace(hour=14, minute=0)
    elif "tối" in time_str:
        dt = dt.replace(hour=18, minute=0)

    hour_match = re.search(r"(\d+)[h|giờ](\d+)?[p|phút]?", time_str)
    if hour_match:
        hour = int(hour_match.group(1))
        minute = int(hour_match.group(2)) if hour_match.group(2) else 0
        dt = dt.replace(hour=hour, minute=minute)

    return dt.isoformat()

# Chuẩn hóa DURATION
def standardize_duration(entities):
    if not entities["DURATION"]:
        return "1 hour"

    duration_str = entities["DURATION"][0]
    match = re.match(r"(\d+)(p|phút|h|giờ|tiếng|ngày)", duration_str)
    if match:
        number = int(match.group(1))
        unit = DURATION_DICT.get(match.group(2), "minute")
        return f"{number} {unit}"
    return "1 hour"

# Chuẩn hóa EVENT
def standardize_event(entities):
    if entities["EVENT"]:
        return entities["EVENT"][0].replace("_", " ")
    return None

@app.post("/schedule")
async def process_text(input: TextInput):
    text = input.text
    timezone = get_localzone()  # Automatically detect server's timezone

    # Bước 1: Nhận diện intent
    intent = detect_intent(text)

    # Bước 2: Trích xuất thực thể
    ner_output = predict_ner(text) if intent else []
    entities = merge_ner_tokens(ner_output)

    # Bước 3: Chuẩn hóa thực thể
    response = {
        "intent": intent,
        "TIME": standardize_time(entities, timezone) if intent else None,
        "EVENT": standardize_event(entities) if intent else None,
        "DURATION": standardize_duration(entities) if intent else None
    }

    return response