import sys
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import unicodedata
import re
import pandas as pd
import json

current_dir = os.path.dirname(os.path.abspath(
    __file__))         # .../ToxicFilter/Module
parent_dir = os.path.abspath(os.path.join(
    current_dir, '..'))    # .../ToxicFilter
sys.path.append(parent_dir)

from Config.VectorComparisonConfig import *


# =====================================================
# 🔹 0. HÀM HỖ TRỢ LOAD MODEL THÔ
# =====================================================

def load_embedding_model(model_path):
    """
    Load mô hình embedding thông minh:
    - Nếu model_path là tên Hugging Face → dùng cache tự động.
    - Nếu model_path là local path → tự tìm thư mục snapshot có model.safetensors hoặc pytorch_model.bin.
    """
    # Nếu là repo name trên Hugging Face (không chứa ổ đĩa, có dạng user/model)
    if not os.path.isabs(model_path):
        print(f"🔹 Loading model from Hugging Face Hub: {model_path}")
        return SentenceTransformer(model_path)

    # Nếu là thư mục local chứa model phẳng
    if any(os.path.isfile(os.path.join(model_path, f)) for f in ["pytorch_model.bin", "model.safetensors"]):
        print(f"🔹 Loading local model from: {model_path}")
        return SentenceTransformer(model_path)

    # Nếu là thư mục local có snapshots
    snapshot_dir = os.path.join(model_path, "snapshots")
    if os.path.exists(snapshot_dir):
        for sub in os.listdir(snapshot_dir):
            sub_path = os.path.join(snapshot_dir, sub)
            if any(os.path.isfile(os.path.join(sub_path, f)) for f in ["pytorch_model.bin", "model.safetensors"]):
                print(f"🔹 Loading model from snapshot: {sub_path}")
                return SentenceTransformer(sub_path)

    # Nếu không tìm thấy
    raise FileNotFoundError(
        f"❌ Không tìm thấy mô hình hợp lệ trong: {model_path}")

# =====================================================
# 🔹 HÀM HỖ TRỢ CHUẨN HÓA VĂN BẢN
# =====================================================


def normalize_text(text):
    # Chuẩn hóa Unicode (đề phòng ký tự “đ”/“ḍ” khác mã)
    text = unicodedata.normalize("NFC", text)

    # Chuyển về chữ thường
    text = text.lower()

    # Loại bỏ ký tự không phải chữ cái, số, khoảng trắng (giữ tiếng Việt có dấu)
    text = re.sub(r"[^a-zA-ZÀ-ỹ0-9\s]", " ", text)

    # Loại bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =====================================================
# 🔹 1. TẠO VÀ LƯU VECTOR DB + FAISS INDEX
# =====================================================


def create_vector_db():
    """
    Tạo và lưu FAISS index từ danh sách cụm từ độc hại.
    Kết quả: vector_db.json + vector_index.faiss
    """
    if VERBOSE:
        print("🔹 Loading embedding model...")
    model = load_embedding_model(MODEL_PATH)

    if VERBOSE:
        print(f"🔹 Reading toxic phrases from {TOXIC_PHRASES_FILE}")

    toxic_phrases = []
    for path in TOXIC_PHRASES_FILE:
        if not os.path.exists(path):
            print(f"⚠️ Warning: File không tồn tại - {path}")
            continue
        with open(path, "r", encoding=ENCODING) as f:
            lines = [
                normalize_text(line.strip())
                for line in f
                if line.strip() and not line.startswith("#")
            ]
            toxic_phrases.extend(lines)

    # --- Loại bỏ trùng lặp ---
    toxic_phrases = list(set(toxic_phrases))

    if VERBOSE:
        print(
            f"🔹 Loaded {len(toxic_phrases)} unique toxic phrases from {len(TOXIC_PHRASES_FILE)} files.")

    if VERBOSE:
        print(f"🔹 Encoding {len(toxic_phrases)} phrases...")
    embeddings = model.encode(toxic_phrases, normalize_embeddings=True)

    # Tạo FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine (do normalize)
    index.add(np.array(embeddings, dtype=np.float32))

    # Ghi file index
    faiss.write_index(index, FAISS_INDEX_FILE)
    if VERBOSE:
        print(f"✅ Saved FAISS index: {FAISS_INDEX_FILE}")

    # Lưu Vector DB JSON
    vector_db = [
        {"text": toxic_phrases[i], "embedding": embeddings[i].tolist()}
        for i in range(len(toxic_phrases))
    ]
    with open(VECTOR_DB_JSON, "w", encoding="utf-8") as f:
        json.dump(vector_db, f, ensure_ascii=False, indent=2)
    if VERBOSE:
        print(f"✅ Saved vector DB JSON: {VECTOR_DB_JSON}")


# =====================================================
# 🔹 2. DÒ TOXIC TRONG FILE EXCEL
# =====================================================

def detect_toxic_from_excel():
    """
    Dò nội dung độc hại trong file Excel đầu vào.
    So sánh embedding của mỗi dòng văn bản với Vector DB bằng FAISS.
    """

    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError(
            "❌ FAISS index not found. Run create_vector_db() first.")

    # --- Load model và FAISS index ---
    model = load_embedding_model(MODEL_PATH)
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(VECTOR_DB_JSON, "r", encoding="utf-8") as f:
        vector_db = json.load(f)
    texts_db = [v["text"] for v in vector_db]

    # --- Đọc dữ liệu Excel ---
    df = pd.read_excel(INPUT_XLSX, nrows=MAX_ROWS)
    if COL_NAME not in df.columns:
        raise ValueError(f"❌ Cột '{COL_NAME}' không tồn tại trong file Excel.")
    texts = df[COL_NAME].astype(str).fillna("").tolist()

    results = []

    if VERBOSE:
        print(f"🔍 Scanning {len(texts)} rows from {INPUT_XLSX}...")

    for i, text in enumerate(texts, start=1):
        text = normalize_text(text)
        emb = model.encode(
            [text], normalize_embeddings=True).astype(np.float32)
        sims, idxs = index.search(emb, k=TOP_K)

        # sim_score = float(sims[0][0])
        # matched_phrase = texts_db[idxs[0][0]]

        toxic_matches = []
        for sim, idx in zip(sims[0], idxs[0]):
            if sim >= THRESHOLD:
                toxic_matches.append({
                    "phrase": texts_db[idx],
                    "similarity": round(float(sim), 3)
                })

        if toxic_matches:
            result = {
                "index": i,
                COL_NAME: text,
                "status": "toxic",
                "toxic_phrase": [m["phrase"] for m in toxic_matches],
                "similarity": [m["similarity"] for m in toxic_matches]
            }
        else:
            result = {
                "index": i,
                COL_NAME: text,
                "status": "clean",
                "toxic_phrase": [],
                "similarity": []
            }

        results.append(result)

    # --- Lưu kết quả ---
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if VERBOSE:
        print(f"✅ Detection complete. Output saved to: {OUTPUT_JSON}")


# =====================================================
# 🔹 3. MAIN TEST
# =====================================================
if __name__ == "__main__":
    print("🔧 Creating FAISS index...")
    create_vector_db()

    print("\n🔍 Detecting toxic posts...")
    detect_toxic_from_excel()
