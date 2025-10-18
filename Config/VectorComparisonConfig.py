import os

# --- Thư mục gốc dự án (ToxicFilter) ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Cấu hình đường dẫn mô hình embedding ---
MODEL_DIR = "Model"
MODEL_PATH = os.path.join(MODEL_DIR, "sup-SimCSE-VietNamese-phobert-base")

# --- File chứa danh sách cụm từ độc hại để build Vector DB ---
TOXIC_PHRASES_FILE = [
    os.path.join(BASE_DIR, "Database", "Phrases.txt"),
    os.path.join(BASE_DIR, "Database", "Keywords.txt")
] 

# --- File FAISS index và JSON vector database ---
FAISS_INDEX_FILE = os.path.join(BASE_DIR, "Database", "Vector", "vector_index.faiss")
VECTOR_DB_JSON = os.path.join(BASE_DIR, "Database", "Vector", "vector_db.json")

# --- File đầu vào và đầu ra ---
INPUT_FILE = os.path.join(BASE_DIR, "Documents", "Văn bản 1.pdf")
OUTPUT_JSON = os.path.join(BASE_DIR, "Output", "VectorComparison.json")

# --- Tham số xử lý ---
THRESHOLD = 0.7      # Ngưỡng tối thiểu của độ tương đồng (similarity) để một văn bản bị xem là “toxic”
TOP_K = 3            # Số cụm gần nhất lấy từ FAISS
VERBOSE = True       # Bật in log chi tiết

# --- Encoding ---
ENCODING = "utf-8"
