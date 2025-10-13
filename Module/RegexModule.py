import sys, os, re
import unicodedata, re
import json

current_dir = os.path.dirname(os.path.abspath(__file__))         # .../ToxicFilter/Module
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))    # .../ToxicFilter
sys.path.append(parent_dir)

from Config.RegexConfig import *

# ============================================================
# 🧩 HÀM HỖ TRỢ CHUẨN HÓA VĂN BẢN
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
# 🧩 1. TẠO PATTERN REGEX
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
# 🧩 2. HÀM KIỂM TRA MỘT VĂN BẢN
# ============================================================

def find_toxic_in_text(text: str, pattern: re.Pattern):
    """
    Tìm và trả về danh sách các cụm từ độc hại trong một văn bản.
    """
    matches = pattern.findall(text.lower())
    return list(set(matches))


# ============================================================
# 🧩 3. HÀM XỬ LÝ TOÀN BỘ FILE EXCEL
# ============================================================

def detect_toxic_from_excel(excel_path: str, col_name: str, max_rows=None):
    """
    Kết hợp toàn bộ pipeline:
    - Đọc file Excel
    - Nạp danh sách từ độc hại
    - Biên dịch regex
    - Trả về danh sách kết quả theo từng dòng
    """
    texts = load_excel_texts(filepath=excel_path, col_name=col_name, max_rows=max_rows)
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
            col_name: text,
            "status": status,
            "toxic_detected": detected if detected else ["none"]
        })
    return results


# ============================================================
# 🧩 4. TEST ĐƠN GIẢN (chạy riêng module)
# ============================================================
if __name__ == "__main__":
    print("🔍 Đang quét dữ liệu Excel...")

    data = detect_toxic_from_excel(INPUT_XLSX, COL_NAME, max_rows=MAX_ROWS)
    total = len(data)
    toxic_count = sum(1 for x in data if x["status"] == "toxic")

    # --- Xuất ra JSON ---
    output_path = OUTPUT_JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Quét hoàn tất! Đã xử lý {total} dòng, phát hiện {toxic_count} toxic ({toxic_count/total:.1%})")
    print(f"💾 Kết quả được lưu tại: {output_path}")
