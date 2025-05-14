from transformers import AutoModelForTokenClassification, AutoTokenizer
import os

def download_model(save_dir='./model'):
    # Kiểm tra các file quan trọng
    required_files = ["added_tokens.json", "bpe.codes", "config.json", "model.safetensors", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"]
    model_ready = all(os.path.exists(os.path.join(save_dir, f)) for f in required_files)

    if model_ready:
        print("Model already exists locally. Skipping download.")
        return

    print("Downloading model from Hugging Face...")
    model = AutoModelForTokenClassification.from_pretrained("trilekhai/phobert-base-finetuned-ner")
    tokenizer = AutoTokenizer.from_pretrained("trilekhai/phobert-base-finetuned-ner")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Model downloaded and saved.")

if __name__ == "__main__":
    download_model()
