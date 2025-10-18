import sys, os, re, unicodedata, json
from docx import Document
import fitz  # PyMuPDF


current_dir = os.path.dirname(os.path.abspath(__file__))         # .../ToxicFilter/Module
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))    # .../ToxicFilter
sys.path.append(parent_dir)

from Config.RegexConfig import *

# ============================================================
# 🧩 1: ĐỌC DỮ LIỆU TỪ FILE (PDF / DOCX / XLSX)
# ============================================================

def load_texts_from_file(filepath, max_pages=10):
    """
    Đọc văn bản từ .pdf, .docx hoặc .xlsx.
    Trả về danh sách các đoạn text.
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
                texts.append(page.get_text().strip())

    elif ext == ".docx":
        doc = Document(filepath)
        texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    else:
        raise ValueError(f"Định dạng file không được hỗ trợ: {ext}")

    if VERBOSE:
        print(f"📘 Đã tải {len(texts)} đoạn văn từ file: {filepath}")

    return texts

# ============================================================
# 🧩 2: ĐỌC DANH SÁCH TỪ KHÓA ĐỘC HẠI
# ============================================================

def load_toxic_keywords(filepath):
    """
    Đọc file chứa danh sách từ khóa độc hại (bỏ qua dòng trống và comment '#').
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    with open(filepath, "r", encoding=ENCODING) as f:
        keywords = [
            line.strip().lower()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if VERBOSE:
        print(f"🔹 Đã tải {len(keywords)} từ khóa độc hại từ: {filepath}")

    return keywords

# ============================================================
# 🧩 3: CHUẨN HÓA VĂN BẢN
# ============================================================

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


# ============================================================
# 🧩 4. TẠO PATTERN REGEX
# ============================================================

def compile_toxic_regex(toxic_words: list[str]):
    """
    Biên dịch danh sách từ khóa thành pattern regex lớn.
    Dùng cờ IGNORECASE để không phân biệt hoa thường.
    """
    if not toxic_words:
        return re.compile(r"$^")  # regex rỗng
    escaped = [re.escape(word) for word in toxic_words]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


# ============================================================
# 🧩 5. HÀM KIỂM TRA VĂN BẢN
# ============================================================

def find_toxic_in_text(text: str, pattern: re.Pattern):
    """
    Tìm và trả về danh sách các cụm từ độc hại trong văn bản.
    """
    matches = pattern.findall(text.lower())
    return list(set(matches))


# ============================================================
# 🧩 6. HÀM XỬ LÝ TOÀN BỘ FILE (PDF / DOCX / XLSX)
# ============================================================

def detect_toxic_from_file(input_path):
    """
    Kết hợp toàn bộ pipeline:
    - Đọc file Excel
    - Nạp danh sách từ độc hại
    - Biên dịch regex
    - Trả về danh sách kết quả theo từng dòng
    """
    texts = load_texts_from_file(filepath=input_path)
    toxic_words = []
    for file_path in TOXIC_PHRASES_FILE:
        toxic_words.extend([normalize_text(w) for w in load_toxic_keywords(file_path)])
    pattern = compile_toxic_regex(toxic_words)

    results = []
    for i, text in enumerate(texts, start=1):
        clean_text = normalize_text(text)
        detected = find_toxic_in_text(clean_text, pattern)
        status = "toxic" if detected else "clean"
        results.append({
            "index": i,
            "text": text,
            "status": status,
            "toxic_detected": detected if detected else ["none"]
        })
    return results


# ============================================================
# 🧩 7. TEST ĐƠN GIẢN (chạy riêng module)
# ============================================================
def mainRun():
    print("🔍 Đang quét dữ liệu Excel...")

    input_path = INPUT_FILE  

    data = detect_toxic_from_file(input_path)
    total = len(data)
    toxic_count = sum(1 for x in data if x["status"] == "toxic")

    # --- Xuất ra JSON ---
    output_path = OUTPUT_JSON
    # with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Quét hoàn tất! Đã xử lý {total} đoạn, phát hiện {toxic_count} toxic ({toxic_count/total:.1%})")
    print(f"💾 Kết quả được lưu tại: {OUTPUT_JSON}")

    return data