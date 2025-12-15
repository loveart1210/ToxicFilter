import os, sys, json, re, unicodedata, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Cho phép import từ thư mục gốc ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Config.SemanticConfigTest import *

# --- Thư mục lưu mô hình ---
MODEL_DIR = r"D:\Model"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# =====================================================
# 🔹 1. CHUẨN HÓA VĂN BẢN
# =====================================================
def normalize_text(text):
    """Chuẩn hóa văn bản tiếng Việt trước khi đưa vào model."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ỹ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =====================================================
# 🔹 2. TẢI MÔ HÌNH LLAMA-3.2-3B-INSTRUCT
# =====================================================
def load_llama_model():
    """Tự động tải hoặc load model Llama-3.2-3B-Instruct từ D:\Model."""
    try:
        print(f"📦 Kiểm tra mô hình trong cache: {MODEL_DIR} ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_DIR,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print(f"✅ Model đã sẵn sàng tại cache: {MODEL_DIR}")
    except Exception as e:
        print(f"⚠️ Không thể load từ cache ({e}). Đang tải từ Internet...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.save_pretrained(os.path.join(MODEL_DIR, "Llama-3.2-3B-Instruct"))
        tokenizer.save_pretrained(os.path.join(MODEL_DIR, "Llama-3.2-3B-Instruct"))
    return tokenizer, model


# =====================================================
# 🔹 3. PHÂN TÍCH NGỮ NGHĨA MỘT CÂU
# =====================================================
def classify_llama(text, generator):
    """Phân tích ngữ nghĩa văn bản bằng Llama-3.2-3B-Instruct."""
    text = normalize_text(text)
    if not text.strip():
        return "CLEAN", "Không có nội dung."

    prompt = f"""
    Hãy phân loại đoạn văn sau thành một trong ba nhãn:
    CLEAN (bình thường, không xúc phạm),
    OFFENSIVE (có ngôn ngữ xúc phạm hoặc thô tục),
    HATE (ngôn ngữ thù ghét hoặc tấn công cá nhân/nhóm).

    Trả về JSON ngắn gọn với 2 trường:
    {{
      "label": "CLEAN|OFFENSIVE|HATE",
      "reason": "giải thích ngắn vì sao"
    }}

    Đoạn văn: "{text}"
    """

    output = generator(prompt, max_new_tokens=200, temperature=0.1, do_sample=False)
    response = output[0]["generated_text"]

    # Cố gắng parse JSON trong output
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        parsed = json.loads(response[json_start:json_end])
        label = parsed.get("label", "CLEAN").upper()
        reason = parsed.get("reason", "").strip()
    except Exception:
        label, reason = "CLEAN", response.strip()

    return label, reason


# =====================================================
# 🔹 4. PHÂN TÍCH FILE EXCEL
# =====================================================
def semantic_analysis_excel(excel_path, col_name):
    """Đọc file Excel, phân tích từng dòng bằng Llama-3.2-3B-Instruct."""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file Excel: {excel_path}")

    df = pd.read_excel(excel_path, nrows=MAX_ROWS)
    if col_name not in df.columns:
        raise ValueError(f"❌ Cột '{col_name}' không tồn tại trong file Excel.")

    texts = df[col_name].astype(str).fillna("").tolist()

    if VERBOSE:
        print(f"📘 Đã tải {len(texts)} dòng từ: {excel_path}")
        print(f"📑 Cột đang xử lý: {col_name}")
        print("🔹 Đang tải mô hình Llama-3.2-3B-Instruct...")

    tokenizer, model = load_llama_model()
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    results = []
    for i, text in enumerate(texts, start=1):
        label, reason = classify_llama(text, generator)
        results.append({
            "index": i,
            col_name: text,
            "semantic_label": label,
            "explanation": reason
        })

        if VERBOSE and i % 10 == 0:
            print(f"  🔹 Đã xử lý {i}/{len(texts)} dòng...")

    return results


# =====================================================
# 🔹 5. CHẠY THỬ VÀ XUẤT JSON
# =====================================================
if __name__ == "__main__":
    print("🔍 Đang phân tích ngữ nghĩa bằng Llama-3.2-3B-Instruct...")

    data = semantic_analysis_excel(INPUT_XLSX, COL_NAME)

    total = len(data)
    hate = sum(1 for x in data if x["semantic_label"] == "HATE")
    offensive = sum(1 for x in data if x["semantic_label"] == "OFFENSIVE")
    clean = total - hate - offensive

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Phân tích hoàn tất {total} dòng.")
    print(f"💾 Kết quả lưu tại: {OUTPUT_JSON}")
    print(f"📊 CLEAN={clean}, OFFENSIVE={offensive}, HATE={hate}")
