import os
import pandas as pd

# ============================================================
# ⚙️ CẤU HÌNH CƠ BẢN
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # thư mục gốc ToxicFilter

# Đường dẫn tới file Excel chứa dữ liệu đầu vào
INPUT_XLSX = os.path.join(BASE_DIR, "Documents", "Confessions of HNMU.xlsx")

# Tên cột chứa văn bản cần quét
COL_NAME = "post_text"

# Đọc bao nhiêu dòng đầu (None = đọc toàn bộ)
MAX_ROWS = 10

# File txt chứa danh sách từ khóa độc hại
TOXIC_PHRASES_FILE = [
    os.path.join(BASE_DIR, "Database", "Keywords.txt"),
    os.path.join(BASE_DIR, "Database", "Phrases.txt")
]

# File JSON xuất kết quả (nếu dùng main.py để lưu)
OUTPUT_JSON = os.path.join(BASE_DIR, "Output", "Regex.json")

# ============================================================
# ⚙️ CÁC THAM SỐ KHÁC
# ============================================================

# Chế độ hiển thị log
VERBOSE = True

# Mã hóa đọc file
ENCODING = "utf-8"

# ============================================================
# 📘 HÀM XỬ LÝ FILE EXCEL
# ============================================================

def load_excel_texts(filepath=INPUT_XLSX, col_name=COL_NAME, max_rows=MAX_ROWS):
    """
    Đọc dữ liệu văn bản từ file Excel.
    Trả về danh sách các chuỗi trong cột được chọn.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file Excel: {filepath}")

    df = pd.read_excel(filepath, nrows=max_rows)
    if col_name not in df.columns:
        raise ValueError(f"Cột '{col_name}' không tồn tại trong file Excel.")

    texts = df[col_name].astype(str).fillna("").tolist()

    if VERBOSE:
        print(f"📘 Đã tải {len(texts)} dòng từ file: {filepath}")
        print(f"📑 Cột sử dụng: {col_name}")

    return texts


# ============================================================
# 📘 HÀM ĐỌC DANH SÁCH TỪ KHÓA ĐỘC HẠI
# ============================================================

def load_toxic_keywords(filepath=TOXIC_PHRASES_FILE):
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
