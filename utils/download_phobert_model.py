from transformers import AutoModelForTokenClassification, AutoTokenizer
import os

def download_model(save_dir='./model'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model = AutoModelForTokenClassification.from_pretrained("trilekhai/phobert-base-finetuned-ner")
    tokenizer = AutoTokenizer.from_pretrained("trilekhai/phobert-base-finetuned-ner")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    download_model()
