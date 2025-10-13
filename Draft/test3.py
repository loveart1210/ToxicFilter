import re
import os
import sys
import json
import torch
import regex
import pandas as pd
import logging
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

print(torch.__version__)
print(torch.cuda.is_available())

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# 1. Danh sách từ khóa toxic (regex)
VI_TOXIC_LEXICON = [
    "địt","đụ","đéo","đếch","đcm","đm","vãi","vkl","vcl","vãi l*n","vãi lồn",
    "đồ ngu","ngu","đần","đần độn","óc chó","óc lợn","óc heo","khốn","khốn nạn",
    "cặn bã","rác rưởi","bẩn thỉu","bẩn bựa","cút","cút mẹ","cút đi",
    "dm","cc","clgt","má mày","mẹ mày","cha mày","bố mày","tao","mày",
    "lồn","loz","cặc","đĩ","con đĩ","điếm","gái điếm","thằng chó","thằng",
    "ngu học","thằng đần","bitch","fuck","wtf","asshole","dick","pussy","bastard"
]

# Compile regex (case-insensitive, có xử lý dấu cách)
toxic_patterns = [re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE) for word in VI_TOXIC_LEXICON]

# 2. Hàm regex check
def detect_toxic_regex(text: str):
    found = []
    for pat, word in zip(toxic_patterns, VI_TOXIC_LEXICON):
        if pat.search(text):
            found.append(word)
    return found

# 3. Load PhoBERT v2
custom_cache_dir = "D:/Model"
os.makedirs(custom_cache_dir, exist_ok=True)

logging.info("⏳ Đang load PhoBERT v2...")
model_name = "vinai/phobert-base-v2"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=custom_cache_dir,
    use_fast=False,          # ⚠️ rất quan trọng với PhoBERT
    local_files_only=False
)

model = AutoModel.from_pretrained(
    model_name,
    cache_dir=custom_cache_dir,
    local_files_only=False
)
model.eval()
logging.info("✅ PhoBERT v2 đã sẵn sàng")

def clean_text(text: str):
    # bỏ emoji, ký tự ngoài BMP (PhoBERT không học)
    text = regex.sub(r"[^\p{Latin}\p{M}\p{Zs}\p{P}\p{N}]", " ", text)
    return text.lower().strip()

# 4. Hàm lấy embedding
def get_embedding(text: str):
    # chuẩn hóa văn bản
    norm_text = clean_text(text)
    # tokenize
    inputs = tokenizer(norm_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1)
    return emb

# Precompute embeddings cho nhãn
toxic_emb = get_embedding("toxic ngôn ngữ độc hại")
nontoxic_emb = get_embedding("non-toxic ngôn ngữ lành mạnh")

def classify_with_phobert(text: str):
    emb = get_embedding(text)
    sim_toxic = F.cosine_similarity(emb, toxic_emb).item()
    sim_non = F.cosine_similarity(emb, nontoxic_emb).item()
    return "toxic" if sim_toxic > sim_non else "non-toxic"

# 5. Hàm kiểm tra toxic
def check_post_toxic(post_text: str, index: int):
    regex_found = detect_toxic_regex(post_text)
    if regex_found:
        logging.info(f"[{index}] Regex phát hiện: {regex_found}")
        return {
            "index": index,
            "post_text": post_text,
            "method": "regex",
            "status": "toxic",
            "toxic_word": regex_found,
        }
    else:
        status = classify_with_phobert(post_text)
        logging.info(f"[{index}] PhoBERT phân loại: {status}")
        return {
            "index": index,
            "post_text": post_text,
            "method": "phobert",
            "status": status,
            "toxic_word": None,
        }
        
if __name__ == "__main__":
    # 1. Nhập đường dẫn file
    input_path = "Confessions of HNMU.xlsx"
    output_path = "output_toxic.json"
    col_name = "post_text"

    # 2. Đọc file Excel
    df = pd.read_excel(input_path)

    if col_name not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{col_name}' trong file Excel.")

    # 3. Xử lý từng post
    results = []
    for idx, text in enumerate(df[col_name].astype(str).tolist()):
        logging.info(f"🔎 Đang xử lý post {idx + 1}/{len(df)}")  # +1 để dễ đọc (1-based index)
        result = check_post_toxic(text, idx)
        results.append(result)

    # 4. Ghi JSON output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("[\n")  # mở đầu JSON array

        for idx, text in enumerate(df[col_name].astype(str).tolist()):
            logging.info(f"🔎 Đang xử lý post {idx+1}/{len(df)}")
            result = check_post_toxic(text, idx)

            # chuyển object sang JSON string
            line = json.dumps(result, ensure_ascii=False, indent=4)

            # thêm dấu phẩy giữa các object
            if idx > 0:
                f.write(",\n")

            f.write(line)
            f.flush()   # đảm bảo ghi ngay ra file, không chờ hết vòng lặp

        f.write("\n]\n")  # kết thúc JSON array

    logging.info(f"✅ Kết quả đã được lưu vào: {output_path}")
