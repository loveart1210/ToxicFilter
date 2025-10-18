from sentence_transformers import SentenceTransformer
import os
import shutil

# 🔧 Buộc Hugging Face chạy ẩn danh, không dùng token
os.environ["HF_HUB_DISABLE_TOKEN"] = "1"

MODEL_NAME = "VoVanPhuc/sup-SimCSE-Vietnamese-phobert-base"
MODEL_DIR = os.path.join("Model")
LOCAL_PATH = os.path.join(MODEL_DIR, "models--" + MODEL_NAME.replace("/", "--"))

# 🧹 Xóa model cũ nếu tồn tại
if os.path.exists(LOCAL_PATH):
    print(f"🧹 Đang xóa model cũ tại: {LOCAL_PATH}")
    shutil.rmtree(LOCAL_PATH)

# ⬇️ Tải model mới
print(f"⬇️ Đang tải mô hình '{MODEL_NAME}' vào {MODEL_DIR} ...")
model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_DIR)
print("✅ Đã tải xong mô hình!")

# 🧩 Xác minh model đã có file trọng số
expected_files = ["pytorch_model.bin", "model.safetensors"]
found = any(os.path.exists(os.path.join(root, f)) 
            for root, _, files in os.walk(LOCAL_PATH) for f in files if f in expected_files)

if found:
    print(f"📁 Mô hình đã lưu tại: {os.path.abspath(LOCAL_PATH)}")
else:
    print("⚠️ Cảnh báo: Model tải xong nhưng chưa có file trọng số. Hãy kiểm tra lại kết nối mạng.")
