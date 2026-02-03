import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ================= 配置区域 =================
DATA_ROOT = '../../data'  # 数据集根目录
OUTPUT_DIR = './pretrained_emb'  # 输出目录
MODEL_NAME = 'all-MiniLM-L6-v2'  # HuggingFace 模型名称
BATCH_SIZE = 128  # 编码时的 Batch Size

# 定义数据集列表
DATASETS = ['dbbook2014', 'book_crossing', 'ml1m']


# ===========================================

def clean_text_dbbook(text):
    """
    处理 dbbook2014: http://dbpedia.org/resource/Dragonfly_in_Amber -> Dragonfly in Amber
    """
    prefix = "http://dbpedia.org/resource/"
    if text.startswith(prefix):
        text = text[len(prefix):]
    # 将下划线替换为空格，并将URL编码字符解码（简单处理）
    text = text.replace('_', ' ').replace('(', '').replace(')', '')
    return text.strip()


def clean_text_ml1m(text):
    """
    处理 ml1m: http://dbpedia.org/resource/Jumanji 或 None
    """
    if text == 'None' or text is None:
        return ""  # 或者返回 "Unknown"

    prefix = "http://dbpedia.org/resource/"
    if text.startswith(prefix):
        text = text[len(prefix):]
    text = text.replace('_', ' ').replace('(', '').replace(')', '')
    return text.strip()


def clean_text_bookcrossing(text):
    """
    处理 book_crossing: 直接是标题，不需要特殊处理，只需去除首尾空格
    """
    return text.strip()


def get_clean_func(dataset_name):
    if dataset_name == 'dbbook2014':
        return clean_text_dbbook
    elif dataset_name == 'ml1m':
        return clean_text_ml1m
    elif dataset_name == 'book_crossing':
        return clean_text_bookcrossing
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_user_texts(file_path):
    print(f"Loading users from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取最大ID以确定List大小
    # 假设 user key 是字符串形式的整数 "0", "1"...
    max_id = max([int(k) for k in data.keys()])
    num_users = max_id + 1

    user_texts = [""] * num_users

    for uid_str, content in data.items():
        uid = int(uid_str)
        prefs = content.get('preferences', [])
        # 将用户的多条偏好合并为一个长字符串
        # 使用 ". " 连接，保持句子结构
        combined_text = ". ".join(prefs)
        user_texts[uid] = combined_text

    return user_texts


def load_item_texts(file_path, dataset_name):
    print(f"Loading items from {file_path}...")
    clean_func = get_clean_func(dataset_name)

    item_map = {}  # remap_id -> text
    max_id = 0

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # 跳过第一行 header
    # 注意：你需要确认第一行是否真的是 header，如果你的文件没有 header，请注释掉下面这行
    # 根据你提供的数据样例，似乎是有 header 的 ("org_id remap_id entity_name")
    header = lines[0].strip()
    start_idx = 0
    if "org_id" in header or "remap_id" in header:
        start_idx = 1

    for line in lines[start_idx:]:
        parts = line.strip().split(maxsplit=2)  # 只分割前两个空格
        if len(parts) < 2:
            continue

        # 根据你提供的格式:
        # dbbook2014: org_id(0) remap_id(1) entity_name(2)
        # ml1m:       org_id(0) remap_id(1) entity_name(2)
        # book_cross: org_id(0) remap_id(1) item_name(2)

        remap_id = int(parts[1])
        if len(parts) > 2:
            raw_text = parts[2]
        else:
            raw_text = ""  # 某些行可能没有名字

        clean_text = clean_func(raw_text)
        item_map[remap_id] = clean_text
        if remap_id > max_id:
            max_id = remap_id

    # 构建列表，确保索引对应 ID
    num_items = max_id + 1
    item_texts = [""] * num_items
    for i in range(num_items):
        if i in item_map:
            item_texts[i] = item_map[i]
        else:
            item_texts[i] = ""  # 缺失物品填充空字符串

    return item_texts


def main():
    # 检查 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载模型
    print(f"Loading SBERT model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for dataset in DATASETS:
        print(f"\nProcessing dataset: {dataset} ===================")
        dataset_path = os.path.join(DATA_ROOT, dataset)

        # 1. 处理用户
        user_json_path = os.path.join(dataset_path, 'user_preferences.json')
        if os.path.exists(user_json_path):
            user_texts = load_user_texts(user_json_path)
            print(f"Encoding {len(user_texts)} users...")

            # 批量编码
            user_emb = model.encode(user_texts, batch_size=BATCH_SIZE, convert_to_tensor=True, show_progress_bar=True)

            # 保存
            save_path = os.path.join(OUTPUT_DIR, f'{dataset}_user_semantic_emb.pt')
            torch.save(user_emb.cpu(), save_path)  # 保存到 CPU 以防加载时显存不足
            print(f"Saved user embeddings to {save_path} (Shape: {user_emb.shape})")
        else:
            print(f"Warning: {user_json_path} not found.")

        # 2. 处理物品
        item_txt_path = os.path.join(dataset_path, 'item_list.txt')
        if os.path.exists(item_txt_path):
            item_texts = load_item_texts(item_txt_path, dataset)
            print(f"Encoding {len(item_texts)} items...")

            # 批量编码
            item_emb = model.encode(item_texts, batch_size=BATCH_SIZE, convert_to_tensor=True, show_progress_bar=True)

            # 保存
            save_path = os.path.join(OUTPUT_DIR, f'{dataset}_item_semantic_emb.pt')
            torch.save(item_emb.cpu(), save_path)
            print(f"Saved item embeddings to {save_path} (Shape: {item_emb.shape})")
        else:
            print(f"Warning: {item_txt_path} not found.")


if __name__ == "__main__":
    main()
