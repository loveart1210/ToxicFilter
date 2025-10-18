import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # thư mục gốc ToxicFilter

# Đường dẫn tới file Excel chứa dữ liệu đầu vào
INPUT_FILE = os.path.join(BASE_DIR, "Documents", "Văn bản 1.pdf")

# File txt chứa danh sách từ khóa độc hại
TOXIC_PHRASES_FILE = [
    os.path.join(BASE_DIR, "Database", "Keywords.txt"),
    os.path.join(BASE_DIR, "Database", "Phrases.txt")
]

# File JSON xuất kết quả (nếu dùng main.py để lưu)
OUTPUT_JSON = os.path.join(BASE_DIR, "Output", "Regex.json")

# Chế độ hiển thị log
VERBOSE = True

# Mã hóa đọc file
ENCODING = "utf-8"
