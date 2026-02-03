"""
噪声数据生成脚本
为训练集生成噪声交互数据
"""
import numpy as np
import os
from collections import defaultdict

# ======================== 配置参数（在开头设置）========================
DATASET_PATH = r"C:\Users\86156\Desktop\1 Projects\Agent模拟噪声\mycode\data\dbbook2014"
NOISE_PER_USER = 5  # 每个用户的噪声样本数
NUM_USERS = None  # 自动从train.txt推断（None表示自动）
NUM_ITEMS = None  # 自动从train.txt推断（None表示自动）


# ======================== 主函数 ========================
def generate_noise_data(dataset_path, noise_per_user, num_users=None, num_items=None):
    """
    为训练集生成噪声数据

    Args:
        dataset_path: 数据集路径
        noise_per_user: 每个用户的噪声样本数
        num_users: 用户总数（None表示自动推断）
        num_items: 物品总数（None表示自动推断）
    """

    train_file = os.path.join(dataset_path, 'train.txt')
    noise_output_file = os.path.join(dataset_path, f'noise_data_{noise_per_user}.txt')

    print("=" * 80)
    print("NOISE DATA GENERATION")
    print("=" * 80)
    print(f"Dataset path: {dataset_path}")
    print(f"Noise per user: {noise_per_user}")
    print(f"Output file: {noise_output_file}")
    print("=" * 80)

    # ======================== 读取训练数据 ========================
    print(f"Reading train data from {train_file}...")

    user_items = defaultdict(set)  # 存储每个用户的正样本物品集合
    max_user_id = -1
    max_item_id = -1
    total_interactions = 0

    with open(train_file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = list(map(int, line.split()))
            user_id = parts[0]
            items = parts[1:]

            user_items[user_id] = set(items)
            max_user_id = max(max_user_id, user_id)
            max_item_id = max(max_item_id, max(items))
            total_interactions += len(items)

    # ======================== 确定用户数和物品数 ========================
    if num_users is None:
        num_users = max_user_id + 1
    if num_items is None:
        num_items = max_item_id + 1

    print(f"Total users: {num_users}")
    print(f"Total items: {num_items}")
    print(f"Total training interactions: {total_interactions}")
    print(f"Average items per user: {total_interactions / len(user_items):.2f}")

    # ======================== 生成噪声数据 ========================
    print(f"\nGenerating noise data ({noise_per_user} per user)...")

    noise_data = {}  # {user_id: [noise_items]}
    generated_count = 0
    failed_count = 0
    max_attempts = 1000  # 最大尝试次数

    for user_id in range(num_users):
        noise_data[user_id] = []
        user_pos_items = user_items.get(user_id, set())

        attempts = 0
        while len(noise_data[user_id]) < noise_per_user and attempts < max_attempts:
            # 随机采样一个物品
            noise_item = np.random.randint(0, num_items)

            # 确保这个物品不在用户的正样本中
            if noise_item not in user_pos_items:
                noise_data[user_id].append(noise_item)
                generated_count += 1

            attempts += 1

        if len(noise_data[user_id]) < noise_per_user:
            failed_count += 1
            print(f"Warning: User {user_id} only got {len(noise_data[user_id])} noise items "
                  f"(expected {noise_per_user})")

    # ======================== 写入噪声数据文件 ========================
    print(f"\nWriting noise data to {noise_output_file}...")

    os.makedirs(dataset_path, exist_ok=True)
    with open(noise_output_file, 'w') as f:
        for user_id in sorted(noise_data.keys()):
            items = noise_data[user_id]
            if len(items) > 0:
                item_str = ' '.join(map(str, sorted(items)))
                f.write(f"{user_id} {item_str}\n")

    # ======================== 统计信息 ========================
    print("\n" + "=" * 80)
    print("NOISE DATA GENERATION COMPLETED")
    print("=" * 80)
    print(f"Total noise interactions generated: {generated_count}")
    print(f"Users with incomplete noise samples: {failed_count}")
    print(f"Output file: {noise_output_file}")
    print("=" * 80)

    # ======================== 验证生成的噪声数据 ========================
    print("\nValidating noise data...")

    noise_user_count = 0
    total_noise_items = 0

    with open(noise_output_file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = list(map(int, line.split()))
            user_id = parts[0]
            items = parts[1:]

            noise_user_count += 1
            total_noise_items += len(items)

            # 验证没有重叠
            user_pos_items = user_items.get(user_id, set())
            noise_items_set = set(items)
            overlap = user_pos_items & noise_items_set

            if len(overlap) > 0:
                print(f"ERROR: User {user_id} has overlap items: {overlap}")

    print(f"Users in noise file: {noise_user_count}")
    print(f"Total noise items: {total_noise_items}")
    print(f"Average noise items per user: {total_noise_items / max(noise_user_count, 1):.2f}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    generate_noise_data(
        dataset_path=DATASET_PATH,
        noise_per_user=NOISE_PER_USER,
        num_users=NUM_USERS,
        num_items=NUM_ITEMS
    )
