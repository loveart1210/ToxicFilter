import re
import os
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU/MPS")

EXCEL_PATH = "HNMU Confessions.xlsx"      # cập nhật đường dẫn
TEXT_COL    = None                           # để None sẽ auto-chọn cột chuỗi đầu tiên
OUT_ROWS    = "toxic_scan_rows.csv"
OUT_VOCAB   = "toxic_terms_found.txt"

# Đường dẫn đến thư mục lưu trữ tùy chỉnh
custom_cache_dir = "D:/Model"  # Thư mục lưu mô hình

# Đảm bảo thư mục tồn tại, nếu không thì tạo mới
os.makedirs(custom_cache_dir, exist_ok=True)

# ===== 1) Lexicon nhanh =====
VI_TOXIC_LEXICON = [
    "địt","đụ","đéo","đếch","đcm","đm","vãi","vkl","vcl","vãi l*n","vãi lồn",
    "đồ ngu","ngu","đần","đần độn","óc chó","óc lợn","óc heo","khốn","khốn nạn",
    "cặn bã","rác rưởi","bẩn thỉu","bẩn bựa","cút","cút mẹ","cút đi",
    "dm","cc","clgt","má mày","mẹ mày","cha mày","bố mày","tao","mày",
    "lồn","loz","cặc","đĩ","con đĩ","điếm","gái điếm","thằng chó","thằng",
    "ngu học","thằng đần","bitch","fuck","wtf","asshole","dick","pussy","bastard"
]

def term_to_regex(term: str) -> str:
    esc = ""
    for ch in term:
        esc += ".{0,3}" if ch == "*" else re.escape(ch)
    esc = esc.replace(r"\ ", r"\s+")
    return r"\b" + esc + r"\b"

PATTERNS = [(t, re.compile(term_to_regex(t), re.I)) for t in VI_TOXIC_LEXICON]

def scan_lexicon(text: str):
    text = text or ""
    tl = text.lower()
    hits = [raw for raw, pat in PATTERNS if pat.search(tl)]
    # loại trùng, giữ thứ tự
    seen, uniq = set(), []
    for h in hits:
        if h not in seen:
            uniq.append(h); seen.add(h)
    return uniq

# ===== 2) LLM Việt (HuggingFace, offline) =====
# Sử dụng mô hình Viet-Mistral/Vistral-7B-Chat với thư mục tùy chỉnh
USE_LLM = True
MODEL_ID = "Viet-Mistral/Vistral-7B-Chat"  # Thay đổi mô hình
LOAD_4BIT = True  # Giảm VRAM (yêu cầu GPU + bitsandbytes)

tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

if USE_LLM:
    kwargs = {}
    if device == "cuda" and LOAD_4BIT:
        kwargs.update(dict(
            device_map="auto",
            load_in_4bit=True
        ))
    else:
        kwargs.update(dict(
            device_map="auto" if device=="cuda" else None,
            torch_dtype=torch.float16 if device=="cuda" else torch.float32
        ))

    # Tải tokenizer và mô hình với cache_dir tùy chỉnh
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=custom_cache_dir,
        local_files_only=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=custom_cache_dir,
        local_files_only=False,
        **kwargs
    )

def llm_extract_toxic(text: str, max_new_tokens=128) -> list:
    if not text or not USE_LLM or model is None:
        return []

    system = (
        "Bạn là bộ lọc ngôn từ độc hại tiếng Việt. "
        "TRẢ VỀ CHỈ danh sách các từ/cụm từ toxic xuất hiện trong văn bản đầu vào, cách nhau bởi dấu phẩy. "
        "Nếu KHÔNG có, trả về đúng từ: None."
    )
    user = f"Văn bản: <<<{text}>>>"

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    # Tạo input đúng định dạng chat cho model hiện tại
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,  # thêm phần bắt đầu của assistant
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen = tokenizer.decode(out[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()

    # xử lý kết quả
    if gen.lower() == "none":
        return []
    # tách danh sách theo dấu phẩy/dòng
    parts = re.split(r"[,;\n]+", gen)
    terms = []
    for p in parts:
        t = p.strip()
        if t and t.lower() != "none":
            terms.append(t)
    # lọc nhiễu: bỏ quá dài (>6 từ) và những câu giải thích
    terms = [t for t in terms if len(t.split()) <= 6]
    # unique, giữ thứ tự
    seen, uniq = set(), []
    for t in terms:
        if t not in seen:
            uniq.append(t); seen.add(t)
    return uniq

# ===== 3) Chạy toàn pipeline =====
df = pd.read_excel(EXCEL_PATH)
if TEXT_COL is None:
    # chọn cột có dtype object đầu tiên
    text_cols = [c for c in df.columns if df[c].dtype == 'object']
    if not text_cols:
        text_cols = [df.columns[0]]
    TEXT_COL = text_cols[0]

results = []
global_set = set()

for text in df[TEXT_COL].astype(str).fillna(""):
    # lexicon trước
    hits = scan_lexicon(text)
    # nếu ít/không có, gọi LLM để bổ sung
    if USE_LLM and len(hits) <= 1:
        llm_hits = llm_extract_toxic(text)
        for h in llm_hits:
            if h not in hits:
                hits.append(h)
    # chuẩn hoá
    hits = [h.strip() for h in hits if h and h.strip()]
    if hits:
        for h in hits:
            global_set.add(h)
        results.append(", ".join(hits))
    else:
        results.append(None)

out = df.copy()
out["toxic_terms"] = results
out.to_csv(OUT_ROWS, index=False, encoding="utf-8")
with open(OUT_VOCAB, "w", encoding="utf-8") as f:
    if global_set:
        f.write("\n".join(sorted(global_set)))
    else:
        f.write("None")

print(f"Đã xuất: {OUT_ROWS}, {OUT_VOCAB}")