import os

# --- Thư mục gốc ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Mô hình ViHateT5 ---
MODEL_NAME = "tarudesu/ViHateT5-base-HSD"

# --- File Excel đầu vào ---
INPUT_XLSX = os.path.join(BASE_DIR, "Documents", "Confessions of HNMU.xlsx")
COL_NAME = "post_text"

# --- File JSON đầu ra ---
OUTPUT_JSON = os.path.join(BASE_DIR, "Output", "Semantic.json")

# --- Giới hạn số dòng đọc (None = toàn bộ) ---
MAX_ROWS = 10

# --- Danh sách nhãn mô hình ---
LABELS = ["CLEAN", "OFFENSIVE", "HATE"]

# --- In log chi tiết ---
VERBOSE = True
