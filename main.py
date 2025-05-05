from fastapi import FastAPI
from pydantic import BaseModel
import py_vncorenlp

import os
import requests

def download_file(url, dst):
    r = requests.get(url, stream=True)
    with open(dst, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def download_model(save_dir='./'):
    if save_dir[-1] == '/':
        save_dir = save_dir[:-1]
    jar_path = os.path.join(save_dir, "VnCoreNLP-1.2.jar")

    if os.path.isdir(save_dir + "/models") and os.path.exists(jar_path):
        print("VnCoreNLP model folder " + save_dir + " already exists! Please load VnCoreNLP from this folder!")
        return

    os.makedirs(save_dir + "/models/dep", exist_ok=True)
    os.makedirs(save_dir + "/models/ner", exist_ok=True)
    os.makedirs(save_dir + "/models/postagger", exist_ok=True)
    os.makedirs(save_dir + "/models/wordsegmenter", exist_ok=True)

    # jar
    download_file("https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.2.jar", jar_path)

    # wordsegmenter
    download_file("https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab",
                  save_dir + "/models/wordsegmenter/vi-vocab")
    download_file("https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr",
                  save_dir + "/models/wordsegmenter/wordsegmenter.rdr")

    # postagger
    download_file("https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/postagger/vi-tagger",
                  save_dir + "/models/postagger/vi-tagger")

    # ner
    download_file("https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/ner/vi-500brownclusters.xz",
                  save_dir + "/models/ner/vi-500brownclusters.xz")
    download_file("https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/ner/vi-ner.xz",
                  save_dir + "/models/ner/vi-ner.xz")
    download_file("https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/ner/vi-pretrainedembeddings.xz",
                  save_dir + "/models/ner/vi-pretrainedembeddings.xz")

    # parse
    download_file("https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/dep/vi-dep.xz",
                  save_dir + "/models/dep/vi-dep.xz")

app = FastAPI()


# Tải mô hình
vncorenlp_path = os.path.join(os.getcwd(), "vncorenlp_wrapper")
download_model(save_dir=vncorenlp_path)

# Khởi tạo mô hình
rdrsegmenter = py_vncorenlp.VnCoreNLP(save_dir=vncorenlp_path, annotators=['wseg'])

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    # Phân tích văn bản
    output = rdrsegmenter.word_segment(input.text)
    return {"output": output}

# Cho phép chỉ chạy download nếu gọi riêng
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "download_only":
        download_model(save_dir=vncorenlp_path)
        print("Model downloaded.")
