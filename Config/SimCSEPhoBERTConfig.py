import os

# --- Thư mục gốc dự án (ToxicFilter) ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Cấu hình đường dẫn mô hình embedding ---
MODEL_DIR = r"D:\Model"
MODEL_PATH = os.path.join(MODEL_DIR, "models--VoVanPhuc--sup-SimCSE-Vietnamese-phobert-base")

# --- File chứa danh sách cụm từ độc hại để build Vector DB ---
TOXIC_PHRASES_FILE = [
    os.path.join(BASE_DIR, "Database", "Phrases.txt"),
    os.path.join(BASE_DIR, "Database", "Keywords.txt")
] 

# --- File FAISS index và JSON vector database ---
FAISS_INDEX_FILE = os.path.join(BASE_DIR, "Database", "Vector", "vector_index.faiss")
VECTOR_DB_JSON = os.path.join(BASE_DIR, "Database", "Vector", "vector_db.json")

# --- File Excel đầu vào & JSON đầu ra ---
INPUT_XLSX = os.path.join(BASE_DIR, "Documents", "Confessions of HNMU.xlsx")
COL_NAME = "post_text"

# Đọc bao nhiêu dòng đầu (None = đọc toàn bộ)
MAX_ROWS = 10

OUTPUT_JSON = os.path.join(BASE_DIR, "Output", "faiss.json")

# --- Tham số xử lý ---
THRESHOLD = 0.7     # Ngưỡng tối thiểu của độ tương đồng (similarity) để một văn bản bị xem là “toxic”
TOP_K = 3            # Lấy top K cụm gần nhất từ FAISS
VERBOSE = True       # Bật in log chi tiết

# --- Encoding ---
ENCODING = "utf-8"

# ==============================================
# ✅ HÀM HỖ TRỢ CHUYỂN ĐƯỜNG DẪN
# ==============================================
def abs_path(*paths):
    """
    Trả về đường dẫn tuyệt đối dựa theo BASE_DIR
    """
    return os.path.join(BASE_DIR, *paths)
