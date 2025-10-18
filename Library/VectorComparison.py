import sys, os, re, numpy as np, unicodedata, json, faiss, pandas as pd
from sentence_transformers import SentenceTransformer
from docx import Document
import fitz # PyMuPDF
from huggingface_hub import login

current_dir = os.path.dirname(os.path.abspath(__file__))         # .../ToxicFilter/Module
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))    # .../ToxicFilter
sys.path.append(parent_dir)

from Config.VectorComparisonConfig import *

# =====================================================
# 🔹 1. LOAD MODEL EMBEDDING
# =====================================================

def load_embedding_model(model_path, model_dir="Model"):
    """
    Tải mô hình embedding:
    - Nếu model_path là repo name Hugging Face → tải về model_dir nếu chưa có.
    - Nếu model_path là đường dẫn local → tải trực tiếp.
    """
        
    # Nếu là đường dẫn tuyệt đối hoặc thư mục local
    if os.path.isdir(model_path):
        print(f"🔹 Loading local model from: {model_path}")
        return SentenceTransformer(model_path)

    # Tên repo Hugging Face
    repo_name = model_path.strip()

    # Đường dẫn local tương ứng (cache)
    local_model_path = os.path.join(model_dir, "models--" + repo_name.replace("/", "--"))

    # Nếu đã tồn tại local model
    if os.path.exists(local_model_path):
        print(f"🔹 Found local model at: {local_model_path}")
        return SentenceTransformer(local_model_path)

    # Nếu chưa có local model → tải về
    print(f"⬇️ Model not found locally. Downloading from Hugging Face: {repo_name}")
    os.makedirs(model_dir, exist_ok=True)
    model = SentenceTransformer(repo_name, cache_folder=model_dir)
    print(f"✅ Model downloaded and cached in: {local_model_path}")
    return model

# =====================================================
# 🔹 2. CHUẨN HÓA VĂN BẢN
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
# 🔹 3. ĐỌC DỮ LIỆU TỪ FILE (PDF / DOCX)
# =====================================================

def load_texts_from_file(filepath, max_pages=10):
    """
    Đọc nội dung văn bản từ file .pdf hoặc .docx.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    texts = []

    if ext == ".pdf":
        with fitz.open(filepath) as doc:
            for i, page in enumerate(doc):
                if max_pages and i >= max_pages:
                    break
                text = page.get_text().strip()
                if text:
                    texts.append(text)

    elif ext == ".docx":
        doc = Document(filepath)
        texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    else:
        raise ValueError(f"❌ Chỉ hỗ trợ .pdf và .docx, không hỗ trợ {ext}")

    if VERBOSE:
        print(f"📘 Đã tải {len(texts)} đoạn văn từ file: {filepath}")

    return texts

# =====================================================
# 🔹 4. TẠO VÀ LƯU VECTOR DB + FAISS INDEX
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
# 🔹 5. DÒ TOXIC TRONG FILE PDF / DOCX
# =====================================================

def detect_toxic_from_file():
    """
    Dò nội dung độc hại trong file PDF/DOCX đầu vào.
    So sánh embedding của mỗi đoạn với Vector DB bằng FAISS.
    """
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError(
            "❌ FAISS index not found. Run create_vector_db() first.")

    model = load_embedding_model(MODEL_PATH)
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(VECTOR_DB_JSON, "r", encoding="utf-8") as f:
        vector_db = json.load(f)
    texts_db = [v["text"] for v in vector_db]

    texts = load_texts_from_file(INPUT_FILE)
    results = []

    if VERBOSE:
        print(f"🔍 Scanning {len(texts)} đoạn văn từ: {INPUT_FILE}")

    for i, text in enumerate(texts, start=1):
        norm_text = normalize_text(text)
        emb = model.encode(
            [norm_text], normalize_embeddings=True).astype(np.float32)
        sims, idxs = index.search(emb, k=TOP_K)

        toxic_matches = []
        for sim, idx in zip(sims[0], idxs[0]):
            if sim >= THRESHOLD:
                toxic_matches.append({
                    "phrase": texts_db[idx],
                    "similarity": round(float(sim), 3)
                })

        result = {
            "index": i,
            "text": text,
            "status": "toxic" if toxic_matches else "clean",
            "toxic_phrase": [m["phrase"] for m in toxic_matches],
            "similarity": [m["similarity"] for m in toxic_matches]
        }
        results.append(result)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if VERBOSE:
        print(f"✅ Detection complete. Output saved to: {OUTPUT_JSON}")


# =====================================================
# 🔹 6. MAIN TEST
# =====================================================
if __name__ == "__main__":
    print("🔧 Creating FAISS index...")
    create_vector_db()

    print("\n🔍 Detecting toxic content...")
    detect_toxic_from_file()
