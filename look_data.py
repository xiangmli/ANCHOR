import json
import os

# 设置文件路径
base_path = r"C:\Users\86156\Desktop\1 Projects\小论文\参考代码\AgentRecBench\process_data\process_data\output_data_all"
sources = ['amazon', 'goodreads', 'yelp']


def read_first_line_from_file(file_path):
    """读取JSON文件的第一行数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                return json.loads(first_line)
            else:
                return None
    except FileNotFoundError:
        print(f"  文件不存在: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"  JSON解析错误: {e}")
        return None


def display_sample_data():
    """读取并显示每个source的user, item, review各一条示例数据"""

    print("=" * 80)
    print("读取各数据集的示例数据（每个数据集各读取user, item, review各一条）")
    print("=" * 80)

    for source in sources:
        print(f"\n{'=' * 80}")
        print(f"数据集: {source.upper()}")
        print(f"{'=' * 80}")

        source_dir = os.path.join(base_path, source)

        if not os.path.exists(source_dir):
            print(f"警告: 目录 {source_dir} 不存在！")
            continue

        # 读取user.json的第一条数据
        print(f"\n【USER 示例】")
        print("-" * 80)
        user_file = os.path.join(source_dir, 'user.json')
        user_data = read_first_line_from_file(user_file)
        if user_data:
            print(json.dumps(user_data, indent=2, ensure_ascii=False))
        else:
            print("  无数据")

        # 读取item.json的第一条数据
        print(f"\n【ITEM 示例】")
        print("-" * 80)
        item_file = os.path.join(source_dir, 'item.json')
        item_data = read_first_line_from_file(item_file)
        if item_data:
            print(json.dumps(item_data, indent=2, ensure_ascii=False))
        else:
            print("  无数据")

        # 读取review.json的第一条数据
        print(f"\n【REVIEW 示例】")
        print("-" * 80)
        review_file = os.path.join(source_dir, 'review.json')
        review_data = read_first_line_from_file(review_file)
        if review_data:
            print(json.dumps(review_data, indent=2, ensure_ascii=False))
        else:
            print("  无数据")

    print("\n" + "=" * 80)
    print("示例数据读取完成！")
    print("=" * 80)


# 执行函数
if __name__ == "__main__":
    display_sample_data()
