from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import re
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import os
import py_vncorenlp

app = FastAPI()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn VnCoreNLP
vncorenlp_path = os.path.join(CURRENT_DIR, "vncorenlp_wrapper")
print("Current working directory:", os.getcwd())

# Khởi tạo VnCoreNLP Segmenter
rdrsegmenter = py_vncorenlp.VnCoreNLP(save_dir=vncorenlp_path, annotators=['wseg'])
print("Current working directory:", os.getcwd())

# Đường dẫn model PhoBERT
HUGGINGFACE_MODEL_PATH = os.path.join(CURRENT_DIR, "model")

# Load NER model và tokenizer
model = AutoModelForTokenClassification.from_pretrained(HUGGINGFACE_MODEL_PATH, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH, use_fast=False, local_files_only=True)
model.eval()

# Khởi tạo pipeline NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Từ điển ánh xạ
TIME_DICT = {
    "chút nữa": {"hour": 0, "minute": 15},
    "xíu nữa": {"hour": 0, "minute": 15},
    "chút_xíu nữa": {"hour": 0, "minute": 15},
    "tí nữa": {"hour": 0, "minute": 15},
    "sáng": {"hour": 8, "minute": 0},
    "chiều": {"hour": 14, "minute": 0},
    "tối": {"hour": 18, "minute": 0},
    "ngày_mai": {"days": 1},
    "mai": {"days": 1},
    "mốt": {"days": 2},
    "hôm_qua": {"days": -1},
    "hôm trước": {"days": -2},
    "hôm_nay": {"days": 0},
    "bữa_nay": {"days": 0},
    "tuần trước": {"days": -7},
    "tuần sau": {"days": 7},
    "tháng trước": {"months": -1},
    "tháng sau": {"months": 1},
    "năm_ngoái": {"years": -1},
    "năm sau": {"years": 1}
}

DURATION_DICT = {
    "p": "minute",
    "phút": "minute",
    "h": "hour",
    "giờ": "hour",
    "tiếng": "hour",
    "ngày": "day"
}

# Ngưỡng score cho NER
NER_SCORE_THRESHOLD = 0.6

# Input model
class TextInput(BaseModel):
    text: str
    current_datetime: str = Field(default="2025-05-05T22:22:22+07:00")  # ISO 8601 format

# Trích xuất thực thể (NER)
def predict_ner(text):
    segmented_sentence = rdrsegmenter.word_segment(text)
    print("Segmented sentence:", segmented_sentence)
    
    results = ner_pipeline(segmented_sentence)
    print("NER results:", results)
    
    flat_results = [item for sublist in results for item in sublist]
    ner_output = [
        {
            "entity": item.get("entity", item.get("entity_group", "")),
            "word": item["word"],
            "score": float(item["score"])
        }
        for item in flat_results if float(item["score"]) >= NER_SCORE_THRESHOLD
    ]
    return ner_output

# Ghép token bị tách (xử lý @@)
def merge_ner_tokens(ner_output):
    merged = []
    buffer = ""
    current_entity = None

    print("\n--- Bắt đầu merge NER tokens ---")
    for item in ner_output:
        entity = item.get("entity_group", item.get("entity", ""))
        word = item["word"]
        print(f"Đang xử lý token: '{word}', entity: {entity}")

        if entity == current_entity and "@@" in word:
            buffer += word.replace("@@", "")
        else:
            if buffer and current_entity is not None:
                print(f"-> Thêm entity đã merge: {current_entity} = '{buffer}'")
                merged.append({"entity": current_entity, "word": buffer})
            buffer = word.replace("@@", "")
            current_entity = entity

    if buffer and current_entity is not None:
        print(f"-> Thêm entity cuối cùng: {current_entity} = '{buffer}'")
        merged.append({"entity": current_entity, "word": buffer})

    print("--- Kết quả merge ---")
    for item in merged:
        print(f"  {item}")
    print("----------------------\n")

    return merged

# Nhóm thực thể theo loại
def group_entities_by_proximity(entities):
    grouped = {"TIME": [], "EVENT": [], "DURATION": []}
    current_entity = None
    buffer = []

    for item in entities:
        entity = item["entity"]
        word = item["word"]

        if entity == current_entity:
            buffer.append(word)
        else:
            if buffer and current_entity:
                entity_type = current_entity.split('-')[1] if '-' in current_entity else current_entity
                grouped[entity_type].append(" ".join(buffer))
            buffer = [word]
            current_entity = entity

    if buffer and current_entity:
        entity_type = current_entity.split('-')[1] if '-' in current_entity else current_entity
        grouped[entity_type].append(" ".join(buffer))

    print("\n--- Kết quả nhóm thực thể ---")
    for key, value in grouped.items():
        print(f"  {key}: {value}")
    return grouped

# Kiểm tra sự tồn tại của TIME và EVENT
def check_time_and_event(entities):
    has_time = len(entities["TIME"]) > 0
    has_event = len(entities["EVENT"]) > 0
    return has_time and has_event

# Chuẩn hóa TIME và kiểm tra quá khứ
def standardize_time(entities, current_datetime_str):
    current_datetime = datetime.fromisoformat(current_datetime_str)
    time_entities = entities["TIME"]

    if not time_entities:
        return {"is_valid": False, "time": current_datetime.isoformat()}

    time_str = " ".join(time_entities).lower()
    dt = current_datetime
    has_afternoon_evening = "chiều" in time_str or "tối" in time_str

    # Áp dụng TIME_DICT
    for key, value in TIME_DICT.items():
        if key in time_str:
            if "years" in value:
                dt = dt.replace(year=dt.year + value["years"])
            if "months" in value:
                dt = dt + timedelta(days=value["months"] * 30)  # Gần đúng 1 tháng = 30 ngày
            if "days" in value:
                dt = dt + timedelta(days=value["days"])
            if "hour" in value or "minute" in value:
                hour = value.get("hour", dt.hour)
                minute = value.get("minute", dt.minute)
                dt = dt.replace(hour=hour, minute=minute)

    # Xử lý giờ cụ thể
    hour_match = re.search(r"(\d+)[h|giờ](\d+)?[p|phút]?", time_str)
    if hour_match:
        hour = int(hour_match.group(1))
        minute = int(hour_match.group(2)) if hour_match.group(2) else 0
        # Nếu giờ < 12 và có "chiều" hoặc "tối", cộng 12 giờ
        if hour < 12 and has_afternoon_evening:
            hour += 12
        dt = dt.replace(hour=hour, minute=minute)

    # Kiểm tra nếu TIME trong quá khứ
    is_valid = dt >= current_datetime

    return {"is_valid": is_valid, "time": dt.isoformat()}

# Chuẩn hóa DURATION
def standardize_duration(entities):
    duration_entities = entities["DURATION"]
    if not duration_entities:
        return "1 hour"

    duration_str = " ".join(duration_entities).lower()
    match = re.search(r"(\d+)\s*(p|phút|h|giờ|tiếng|ngày)", duration_str)
    if match:
        number = int(match.group(1))
        unit = DURATION_DICT.get(match.group(2), "minute")
        return f"{number} {unit}"
    return "1 hour"

# Chuẩn hóa EVENT
def standardize_event(entities):
    event_entities = entities["EVENT"]
    return " ".join(event_entities) if event_entities else None

@app.post("/schedule")
async def process_text(input: TextInput):
    text = input.text
    current_datetime = input.current_datetime

    # Bước 1: Trích xuất thực thể
    raw_ner = predict_ner(text)
    merged_ner = merge_ner_tokens(raw_ner)
    grouped_ner = group_entities_by_proximity(merged_ner)

    # Kiểm tra sự tồn tại của TIME và EVENT
    if not check_time_and_event(grouped_ner):
        return {
            "intent": False,
            "TIME": None,
            "EVENT": None,
            "DURATION": None
        }

    # Bước 2: Chuẩn hóa thực thể
    time_result = standardize_time(grouped_ner, current_datetime)

    # Kiểm tra nếu TIME trong quá khứ
    if not time_result["is_valid"]:
        return {
            "intent": False,
            "TIME": None,
            "EVENT": None,
            "DURATION": None
        }

    response = {
        "intent": True,
        "TIME": time_result["time"],
        "EVENT": standardize_event(grouped_ner),
        "DURATION": standardize_duration(grouped_ner)
    }

    return response