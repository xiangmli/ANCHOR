import json
import os
import random
import shutil
from collections import defaultdict

# 设置文件路径
base_path = r"C:\Users\86156\Desktop\1 Projects\小论文\参考代码\AgentRecBench\process_data\process_data\output_data_all"
sources = ['amazon', 'goodreads', 'yelp']


def sample_negative_items(source_dir, source_name, K=20, alpha=0.8):
    """
    为每个用户的每个正样本采样K-1个负样本
    alpha比例的交互作为history，剩余1-alpha作为测试集
    """
    print(f"\n正在处理 {source_name} 数据集...")

    user_file = os.path.join(source_dir, 'sample_user.json')
    item_file = os.path.join(source_dir, 'item.json')
    review_file = os.path.join(source_dir, 'sample_review.json')

    # 0. 清除之前的candidate_sets目录
    output_base_dir = os.path.join(source_dir, 'candidate_sets')
    if os.path.exists(output_base_dir):
        print(f"  清除旧的candidate_sets目录...")
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)

    # 1. 读取所有物品ID
    print(f"  正在读取所有物品...")
    all_item_ids = set()
    with open(item_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            item_id = item.get('item_id')
            if item_id:
                all_item_ids.add(item_id)

    print(f"  总物品数: {len(all_item_ids)}")

    # 2. 读取采样的用户
    print(f"  正在读取采样的用户...")
    sampled_users = []
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            user = json.loads(line.strip())
            sampled_users.append(user)

    print(f"  采样用户数: {len(sampled_users)}")

    # 3. 统计每个用户的交互物品（正样本）
    print(f"  正在统计用户交互物品...")
    user_interactions = defaultdict(list)  # user_id -> list of item_ids

    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line.strip())
            user_id = review.get('user_id')
            item_id = review.get('item_id')
            if user_id and item_id:
                user_interactions[user_id].append(item_id)

    # 4. 为每个用户创建文件夹并生成候选集
    print(f"  正在生成负样本候选集 (alpha={alpha}, K={K})...")

    total_samples = 0
    total_history_items = 0

    for user_idx, user in enumerate(sampled_users):
        user_id = user.get('user_id')
        if not user_id:
            continue

        # 创建用户文件夹
        user_dir = os.path.join(output_base_dir, str(user_idx))
        os.makedirs(user_dir, exist_ok=True)

        # 获取该用户的所有交互物品
        all_interactions = user_interactions.get(user_id, [])

        if len(all_interactions) == 0:
            print(f"  警告: 用户{user_idx}没有交互记录")
            continue

        # 计算划分点
        num_history = int(len(all_interactions) * alpha)

        # 划分为history和test
        history_items = all_interactions[:num_history]
        test_items = all_interactions[num_history:]

        # 保存history
        history_record = {
            'user_id': user_id,
            'history_list': history_items
        }
        history_file = os.path.join(user_dir, 'history.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_record, f, ensure_ascii=False, indent=2)

        total_history_items += len(history_items)

        # 计算负样本池（排除所有交互过的物品）
        all_interacted = set(all_interactions)
        negative_pool = list(all_item_ids - all_interacted)

        if len(negative_pool) < K - 1:
            print(f"  警告: 用户{user_idx}的负样本池({len(negative_pool)})小于K-1({K - 1})")

        # 为每个测试集正样本生成候选集
        for sample_idx, ground_truth_item in enumerate(test_items):
            # 采样K-1个负样本
            num_negatives = min(K - 1, len(negative_pool))
            negative_samples = random.sample(negative_pool, num_negatives)

            # 构建候选列表（K-1个负样本 + 1个正样本）
            candidate_list = negative_samples + [ground_truth_item]

            # 打乱候选列表
            random.shuffle(candidate_list)

            # 创建数据记录
            record = {
                'user_id': user_id,
                'candidate_list': candidate_list,
                'ground_truth': ground_truth_item
            }

            # 保存到文件
            output_file = os.path.join(user_dir, f'sample_{sample_idx}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            total_samples += 1

        if (user_idx + 1) % 10 == 0:
            print(f"    已处理 {user_idx + 1} 个用户, 生成 {total_samples} 个样本文件")

    avg_history = total_history_items / len(sampled_users) if sampled_users else 0
    avg_test = total_samples / len(sampled_users) if sampled_users else 0

    print(f"  完成! 共生成 {total_samples} 个候选集文件")
    print(f"  平均每用户历史交互数: {avg_history:.2f}")
    print(f"  平均每用户测试样本数: {avg_test:.2f}")
    print(f"  文件保存在: {output_base_dir}")

    return {
        'users': len(sampled_users),
        'total_samples': total_samples,
        'total_history_items': total_history_items,
        'avg_history': avg_history,
        'avg_test': avg_test,
        'K': K,
        'alpha': alpha
    }


# 设置随机种子以保证可重复性
random.seed(42)

print("=" * 80)
print("开始为每个用户生成history和负样本候选集")
print("=" * 80)

# 对每个数据集进行处理
K_candidate = 20  # 候选集大小（包含1个正样本和K-1个负样本）
alpha_split = 0.8  # history比例
sample_stats = {}

for source in sources:
    source_dir = os.path.join(base_path, source)
    if os.path.exists(source_dir):
        stats = sample_negative_items(source_dir, source, K=K_candidate, alpha=alpha_split)
        sample_stats[source] = stats
    else:
        print(f"\n警告: 目录 {source_dir} 不存在！")

# 打印汇总信息
print("\n" + "=" * 80)
print("负样本生成完成汇总")
print("=" * 80)
for source, stats in sample_stats.items():
    print(f"\n{source.upper()}:")
    print(f"  用户数: {stats['users']}")
    print(f"  历史交互总数: {stats['total_history_items']}")
    print(f"  平均每用户历史交互数: {stats['avg_history']:.2f}")
    print(f"  测试样本文件数: {stats['total_samples']}")
    print(f"  平均每用户测试样本数: {stats['avg_test']:.2f}")
    print(f"  候选集大小K: {stats['K']}")
    print(f"  历史比例alpha: {stats['alpha']}")

print("\n所有候选集文件已生成完成！")
