"""
数据清洗脚本：使用训练好的NoiseDiscriminator模型清洗训练数据
"""
import torch
import numpy as np
import world
import utils
import register
from register import dataset
import Procedure
from os.path import join

# ======================== 配置 ========================
THRESHOLD = 0.5  # 噪声分数阈值，可以调整
MODEL_WEIGHT_PATH = utils.getFileName()  # 加载模型权重路径
OUTPUT_CLEAN_DATA_PATH = world.data_path + '/clean_data.txt'  # 输出路径


def main():
    world.cprint("=" * 80)
    world.cprint("DATA CLEANING WITH NOISE DISCRIMINATOR")
    world.cprint("=" * 80)

    # ======================== 初始化模型 ========================
    world.cprint(f"Loading model from {MODEL_WEIGHT_PATH}")
    model = register.MODELS[world.model_name](world.config, dataset)
    model = model.to(world.device)

    try:
        model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=torch.device('cpu')))
        world.cprint("Model loaded successfully")
    except FileNotFoundError:
        world.cprint(f"ERROR: Model file {MODEL_WEIGHT_PATH} not found!")
        return

    # ======================== 预测噪声分数 ========================
    world.cprint(f"Using threshold: {THRESHOLD}")
    clean_data_dict, noise_stats = Procedure.predict_noise_score(
        dataset,
        model,
        output_file=OUTPUT_CLEAN_DATA_PATH,
        threshold=THRESHOLD
    )

    # ======================== 打印统计信息 ========================
    Procedure.print_noise_statistics(noise_stats)

    # ======================== 绘制分布图（可选） ========================
    try:
        plot_path = world.data_path + '/noise_score_distribution.png'
        Procedure.plot_noise_score_distribution(noise_stats, save_path=plot_path)
    except Exception as e:
        world.cprint(f"Warning: Could not generate plot: {e}")

    # ======================== 输出噪声分数分析 ========================
    world.cprint("=" * 80)
    world.cprint("NOISE SCORE ANALYSIS")
    world.cprint("=" * 80)

    scores = np.array(list(noise_stats['noise_scores'].values()))
    world.cprint(f"Noise score statistics:")
    world.cprint(f"  Min: {scores.min():.6f}")
    world.cprint(f"  Max: {scores.max():.6f}")
    world.cprint(f"  Mean: {scores.mean():.6f}")
    world.cprint(f"  Median: {np.median(scores):.6f}")
    world.cprint(f"  Std: {scores.std():.6f}")
    world.cprint("=" * 80)

    world.cprint(f"Clean data saved to: {OUTPUT_CLEAN_DATA_PATH}")


if __name__ == '__main__':
    main()
