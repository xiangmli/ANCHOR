"""
噪声数据生成脚本
为训练集生成噪声交互数据
根据用户原有交互数的百分比添加噪声
"""
import numpy as np
import os
from collections import defaultdict

# ======================== 配置参数（在开头设置）========================
DATASET_PATH = r"C:\Users\86156\Desktop\1 Projects\Agent模拟噪声\mycode\data\dbbook2014"
NOISE_RATIO = 0.3  # 噪声比例，0.3表示添加30%的噪声
# 比如用户有10个交互，就添加3个噪声
NUM_USERS = None  # 自动从train.txt推断（None表示自动）
NUM_ITEMS = None  # 自动从train.txt推断（None表示自动）


# ======================== 主函数 ========================
def generate_noise_data_by_ratio(dataset_path, noise_ratio, num_users=None, num_items=None):
    """
    为训练集生成噪声数据，按照用户原有交互数的百分比添加

    Args:
        dataset_path: 数据集路径
        noise_ratio: 噪声比例，范围[0, 1]
                    例如：0.3表示添加30%的噪声
        num_users: 用户总数（None表示自动推断）
        num_items: 物品总数（None表示自动推断）
    """

    train_file = os.path.join(dataset_path, 'train.txt')
    noise_output_file = os.path.join(dataset_path, f'noise_data_{int(noise_ratio * 100)}.txt')

    print("=" * 80)
    print("NOISE DATA GENERATION (BY RATIO)")
    print("=" * 80)
    print(f"Dataset path: {dataset_path}")
    print(f"Noise ratio: {noise_ratio:.1%}")
    print(f"Output file: {noise_output_file}")
    print("=" * 80)

    # ======================== 读取训练数据 ========================
    print(f"\nReading train data from {train_file}...")

    user_items = defaultdict(set)  # 存储每个用户的正样本物品集合
    user_item_counts = defaultdict(int)  # 存储每个用户的交互数
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
            user_item_counts[user_id] = len(items)
            max_user_id = max(max_user_id, user_id)
            if len(items) > 0:
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
    print(f"\nGenerating noise data (ratio: {noise_ratio:.1%})...")

    noise_data = {}  # {user_id: [noise_items]}
    generated_count = 0
    failed_count = 0
    total_planned_noise = 0

    # 统计信息
    user_noise_stats = []  # 用于统计每个用户的噪声数量

    for user_id in range(num_users):
        noise_data[user_id] = []
        user_pos_items = user_items.get(user_id, set())

        # 计算该用户应该生成的噪声数量
        num_pos_items = user_item_counts.get(user_id, 0)
        noise_per_user = max(1, int(np.round(num_pos_items * noise_ratio)))  # 至少生成1个噪声

        total_planned_noise += noise_per_user

        # 生成噪声样本
        attempts = 0
        max_attempts = 1000

        while len(noise_data[user_id]) < noise_per_user and attempts < max_attempts:
            # 随机采样一个物品
            noise_item = np.random.randint(0, num_items)

            # 确保这个物品不在用户的正样本中，且没有被重复添加
            if noise_item not in user_pos_items and noise_item not in noise_data[user_id]:
                noise_data[user_id].append(noise_item)
                generated_count += 1

            attempts += 1

        # 记录统计信息
        if num_pos_items > 0:
            user_noise_stats.append({
                'user_id': user_id,
                'pos_items': num_pos_items,
                'planned_noise': noise_per_user,
                'actual_noise': len(noise_data[user_id]),
                'success': len(noise_data[user_id]) == noise_per_user
            })

        if len(noise_data[user_id]) < noise_per_user:
            failed_count += 1

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
    print(f"Noise ratio: {noise_ratio:.1%}")
    print(f"Total planned noise interactions: {total_planned_noise}")
    print(f"Total actual noise interactions: {generated_count}")
    print(f"Success rate: {generated_count / total_planned_noise * 100:.2f}%")
    print(f"Users with incomplete noise samples: {failed_count}")
    print(f"Output file: {noise_output_file}")
    print("=" * 80)

    # ======================== 详细统计 ========================
    print("\n" + "=" * 80)
    print("DETAILED STATISTICS")
    print("=" * 80)

    successful_users = sum(1 for stat in user_noise_stats if stat['success'])
    total_users_with_items = len(user_noise_stats)

    print(f"Users with positive interactions: {total_users_with_items}")
    print(f"Users with successful noise generation: {successful_users}")
    print(f"Users with incomplete noise: {failed_count}")

    # 计算平均值
    if user_noise_stats:
        avg_pos_items = np.mean([stat['pos_items'] for stat in user_noise_stats])
        avg_planned_noise = np.mean([stat['planned_noise'] for stat in user_noise_stats])
        avg_actual_noise = np.mean([stat['actual_noise'] for stat in user_noise_stats])

        print(f"\nAverage positive items per user: {avg_pos_items:.2f}")
        print(f"Average planned noise per user: {avg_planned_noise:.2f}")
        print(f"Average actual noise per user: {avg_actual_noise:.2f}")

    print("=" * 80)

    # ======================== 验证生成的噪声数据 ========================
    print("\nValidating noise data...")

    noise_user_count = 0
    total_noise_items = 0
    overlap_count = 0

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
                overlap_count += 1

    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"Users in noise file: {noise_user_count}")
    print(f"Total noise items: {total_noise_items}")
    print(f"Average noise items per user: {total_noise_items / max(noise_user_count, 1):.2f}")
    print(f"Users with overlap (ERROR): {overlap_count}")
    print("=" * 80 + "\n")

    # ======================== 示例展示 ========================
    print("EXAMPLE: Sample noise data for first 5 users")
    print("=" * 80)

    example_count = 0
    for user_id in sorted(user_items.keys()):
        if example_count >= 5:
            break

        pos_items = sorted(list(user_items[user_id]))
        noise_items = sorted(noise_data.get(user_id, []))
        noise_percent = (len(noise_items) / len(pos_items) * 100) if len(pos_items) > 0 else 0

        print(f"\nUser {user_id}:")
        print(f"  Positive items ({len(pos_items)}): {pos_items[:10]}{'...' if len(pos_items) > 10 else ''}")
        print(f"  Noise items ({len(noise_items)}, {noise_percent:.1f}%): {noise_items}")

        example_count += 1

    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    generate_noise_data_by_ratio(
        dataset_path=DATASET_PATH,
        noise_ratio=NOISE_RATIO,
        num_users=NUM_USERS,
        num_items=NUM_ITEMS
    )
