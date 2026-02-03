import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import os
from model import NoiseDiscriminator

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

# Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = NoiseDiscriminator(world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

print("="*80)
print("DATASET COMPOSITION")
print("="*80)

noise_labels = dataset.getNoiseLabels()
clean_count = np.sum(noise_labels == 0)
noise_count = np.sum(noise_labels == 1)
total = len(noise_labels)

print(f"Total interactions: {total}")
print(f"Clean interactions: {clean_count} ({clean_count/total*100:.2f}%)")
print(f"Noise interactions: {noise_count} ({noise_count/total*100:.2f}%)")
print(f"Ratio (noise/clean): {noise_count/clean_count:.4f}")
print("="*80)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

# ======================== 检查是否使用噪声标签学习 ========================
use_noise_labels = world.config.get('use_noise_labels', False)

if use_noise_labels:
    world.cprint("=" * 80)
    world.cprint("TRAINING WITH NOISE LABEL LEARNING")
    world.cprint("=" * 80)
    cprint(f"Noise MLP layers: {world.config.get('noise_mlp_layers', [256, 128, 1])}")
    cprint(f"Noise loss weight: {world.config.get('noise_loss_weight', 0.5)}")
    cprint(f"Noise MLP dropout: {world.config.get('noise_mlp_dropout', 0.1)}")
    world.cprint("=" * 80)

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()

        # ======================== 测试阶段 ========================
        if epoch % 10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])

        # ======================== 训练阶段 ========================
        if use_noise_labels:
            # 使用噪声标签学习的训练
            output_information = Procedure.BPR_train_with_noise(
                dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w
            )
        else:
            # 原始BPR训练
            output_information = Procedure.BPR_train_original(
                dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w
            )

        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)

finally:
    if world.tensorboard:
        w.close()

# ======================== 训练完成后的数据清洗 ========================
if world.config.get('use_noise_labels', False) and world.config.get('clean_data_after_training', True):
    world.cprint("\n" + "=" * 80)
    world.cprint("STARTING DATA CLEANING AFTER TRAINING")
    world.cprint("=" * 80)

    threshold = world.config.get('noise_filtering_threshold', 0.5)

    # 获取数据集路径
    dataset_path = dataset.data_path

    # 重新加载模型
    Recmodel = Recmodel.eval()

    # 执行数据清洗
    clean_data_dict, noise_stats = Procedure.predict_noise_score(
        dataset,
        Recmodel,
        dataset_path=dataset_path,
        threshold=threshold
    )

    # 打印统计信息
    Procedure.print_noise_statistics(noise_stats)

    # 尝试绘制图表
    try:
        plot_path = os.path.join(dataset_path, 'noise_score_distribution.png')
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

    output_clean_file = os.path.join(dataset_path, 'clean_data.txt')
    world.cprint(f"Clean data saved to: {output_clean_file}")
    world.cprint("=" * 80 + "\n")

# ======================== 保存最终模型参数 ========================
try:
    # 获取保存路径，例如 ./weights/ml1m/
    dataset_name = world.dataset if hasattr(world, "dataset") else dataset.__class__.__name__
    save_dir = os.path.join("weights", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # 文件名，例如 ./weights/ml1m/final_noise_discriminator.pth
    final_weight_path = os.path.join(save_dir, "final_noise_discriminator.pth")

    # 保存参数（CPU-friendly）
    torch.save(Recmodel.state_dict(), final_weight_path)

    world.cprint("=" * 80)
    world.cprint(f"Final model weights saved to:\n  {final_weight_path}")
    world.cprint("=" * 80)

except Exception as e:
    world.cprint(f"ERROR: Failed to save final model weights: {e}")

world.cprint("Training and data cleaning completed!")
