from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timedelta
import re
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from tzlocal import get_localzone
import os
from utils.download_vncorenlp_model import download_model
import py_vncorenlp

app = FastAPI()

vncorenlp_path = os.path.join(os.getcwd(), "vncorenlp_wrapper")
# download_model(save_dir=vncorenlp_path)

# Khởi tạo mô hình
rdrsegmenter = py_vncorenlp.VnCoreNLP(save_dir=vncorenlp_path, annotators=['wseg'])

# Đường dẫn model PhoBERT
HUGGINGFACE_MODEL_PATH = "local/path"

# Load NER model and tokenizer từ Hugging Face
model = AutoModelForTokenClassification.from_pretrained(HUGGINGFACE_MODEL_PATH, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH, use_fast=False, local_files_only=True) 
model.eval()

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
    segmented_sentence = rdrsegmenter.word_segment(text)
    print("Segmented sentence:", segmented_sentence)
    
    # Run NER pipeline
    results = ner_pipeline(segmented_sentence)
    print("NER results:", results)
    
    # Format output to match merge_ner_tokens expectations
    flat_results = [item for sublist in results for item in sublist]

    ner_output = [
        {
            "entity": item.get("entity", item.get("entity_group", "")),
            "word": item["word"],
            "score": float(item["score"])
        }
        for item in flat_results
    ]

    return ner_output

def merge_ner_tokens(ner_output):
    merged = []
    buffer = ""
    current_entity = None

    print("\n--- Bắt đầu merge NER tokens ---")
    for item in ner_output:
        entity = item.get("entity_group", item.get("entity", ""))
        word = item["word"]
        print(f"Đang xử lý token: '{word}', entity: {entity}")

        # Nếu từ trước có dấu @@, thì nối tiếp, còn không thì kết thúc đoạn trước
        if entity == current_entity and "@@" in word:
            buffer += word.replace("@@", "")
        else:
            if buffer and current_entity is not None:
                print(f"-> Thêm entity đã merge: {current_entity} = '{buffer}'")
                merged.append({"entity": current_entity, "word": buffer})
            buffer = word.replace("@@", "")
            current_entity = entity

    # Thêm cái cuối nếu có
    if buffer and current_entity is not None:
        print(f"-> Thêm entity cuối cùng: {current_entity} = '{buffer}'")
        merged.append({"entity": current_entity, "word": buffer})

    print("--- Kết quả merge ---")
    for item in merged:
        print(f"  {item}")
    print("----------------------\n")

    return merged

def group_entities_by_proximity(entities):
    grouped = []
    buffer = []
    current_entity = None

    for item in entities:
        entity = item["entity"]
        word = item["word"]

        if entity == current_entity:
            buffer.append(word)
        else:
            if buffer:
                grouped.append({"entity": current_entity, "word": " ".join(buffer)})
            buffer = [word]
            current_entity = entity

    # Thêm phần cuối
    if buffer:
        grouped.append({"entity": current_entity, "word": " ".join(buffer)})

    print("\n--- Kết quả nhóm thực thể ---")
    for item in grouped:
        print(f"  {item}")
    return grouped

# Chuẩn hóa TIME
def standardize_time(ner_output, timezone):
    current_datetime = datetime.now(tz=timezone)

    # Lọc các thực thể TIME
    time_entities = [item["word"] for item in ner_output if "TIME" in item["entity"].upper()]
    if not time_entities:
        return (current_datetime + timedelta(hours=1)).isoformat()

    time_str = " ".join(time_entities).lower()
    dt = current_datetime

    # Ngữ cảnh ngày
    if "mai" in time_str or "ngày_mai" in time_str:
        dt += timedelta(days=1)
    elif "mốt" in time_str:
        dt += timedelta(days=2)

    # Ngữ cảnh buổi
    if "sáng" in time_str:
        dt = dt.replace(hour=8, minute=0)
    elif "chiều" in time_str:
        dt = dt.replace(hour=14, minute=0)
    elif "tối" in time_str:
        dt = dt.replace(hour=18, minute=0)

    # Chuẩn hóa giờ và phút từ text
    hour = None
    minute = None

    # Bắt giờ
    hour_match = re.search(r"(\d+)\s*(h|giờ)", time_str)
    if hour_match:
        hour = int(hour_match.group(1))

    # Bắt phút
    minute_match = re.search(r"(\d+)\s*(p|phút)", time_str)
    if minute_match:
        minute = int(minute_match.group(1))

    # Nếu có thông tin thì cập nhật, ưu tiên ghép vào giờ mặc định
    if hour is not None or minute is not None:
        dt = dt.replace(
            hour=hour if hour is not None else dt.hour,
            minute=minute if minute is not None else 0
        )

    return dt.isoformat()

# Chuẩn hóa DURATION
def standardize_duration(ner_output):
    # Lọc các thực thể thuộc loại DURATION
    duration_entities = [item["word"] for item in ner_output if "DURATION" in item["entity"].upper()]
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
def standardize_event(ner_output):
    event_entities = [item["word"] for item in ner_output if "EVENT" in item["entity"].upper()]
    if not event_entities:
        return None
    return " ".join(event_entities)

@app.post("/schedule")
async def process_text(input: TextInput):
    text = input.text
    timezone = get_localzone()  # Automatically detect server's timezone

    # Bước 1: Nhận diện intent
    intent = detect_intent(text)

    # Bước 2: Trích xuất thực thể
    raw_ner = predict_ner(text)
    merged_ner = merge_ner_tokens(raw_ner)
    grouped_ner = group_entities_by_proximity(merged_ner)

    # Bước 3: Chuẩn hóa thực thể
    response = {
        "intent": intent,
        "TIME": standardize_time(grouped_ner, timezone) if intent else None,
        "EVENT": standardize_event(grouped_ner) if intent else None,
        "DURATION": standardize_duration(grouped_ner) if intent else None
    }

    return response

# Cho phép chỉ chạy download nếu gọi riêng
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "download_only":
        download_model(save_dir=vncorenlp_path)
        print("VncoreNLP Model downloaded.")
