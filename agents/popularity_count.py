from collections import Counter
from pathlib import Path
import os


def compute_item_popularity(dataset_name: str) -> None:
    """
    根据 train.txt 统计每个物品的流行度（被交互次数），并保存到：
        data/{dataset_name}/item_popularity.txt

    输入格式（train.txt 示例）：
        0 2266 2075 2482 93 ...
        1 390 2072 2071 307 ...
        ...

    输出格式（item_popularity.txt）：
        item_id popularity_count
        0 123
        1 89
        2 0    # 如果你想写出所有物品，也可以补0

    Args:
        dataset_name: 数据集名称，例如 'dbbook2014'
    """
    # 项目根目录：.../mycode/
    dataset_dir = Path(r"/data/ml1m")

    train_file = dataset_dir / "train.txt"
    output_file = dataset_dir / "item_popularity.txt"

    if not train_file.exists():
        raise FileNotFoundError(f"train.txt not found: {train_file}")

    print(f"Reading train file from: {train_file}")

    popularity_counter = Counter()

    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            # 忽略第一个 user_id，统计 item_id
            for token in parts[1:]:
                try:
                    item_id = int(token)
                    popularity_counter[item_id] += 1
                except ValueError:
                    # 防御式处理
                    continue

    # 按 item_id 排序写出
    os.makedirs(dataset_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item_id in sorted(popularity_counter.keys()):
            f.write(f"{item_id} {popularity_counter[item_id]}\n")

    print(f"✓ Saved item popularity to: {output_file}")
    print(f"  Total items counted: {len(popularity_counter)}")


if __name__ == "__main__":
    # 举例：dbbook2014
    compute_item_popularity("dbbook2014")
