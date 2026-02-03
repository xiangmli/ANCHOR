
from utils import timer

import model
import multiprocessing

import os

CORES = multiprocessing.cpu_count() // 2


def BPR_train_with_noise(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    """
    包含噪声标签学习的BPR训练
    噪声分类损失为主，BPR损失为辅助
    """
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    # ======================== 采样 ========================
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)

    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    # ======================== 获取噪声标签 ========================
    noise_labels = torch.Tensor(
        dataset.getNoiseLabelByUserItem(S[:, 0], S[:, 1])
    ).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    noise_labels = noise_labels.to(world.device)

    users, posItems, negItems, noise_labels = utils.shuffle(
        users, posItems, negItems, noise_labels
    )

    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    aver_bpr_loss = 0.
    aver_noise_loss = 0.

    # ======================== 损失权重配置 ========================
    # 噪声损失权重（主损失）
    bpr_loss_weight = world.config.get('bpr_loss_weight', 0.1)  # BPR为辅助，权重较小
    # 噪声损失权重（主损失），不再需要单独配置
    use_noise_labels = world.config.get('use_noise_labels', True)

    # ======================== 批处理训练 ========================
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg,
          batch_noise_labels)) in enumerate(utils.minibatch(
        users, posItems, negItems, noise_labels,
        batch_size=world.config['bpr_batch_size']
    )):

        # ======================== 噪声分类损失（主损失）========================
        if use_noise_labels and Recmodel.noise_mlp is not None:
            noise_loss = Recmodel.noise_classification_loss(
                batch_users, batch_pos, batch_noise_labels
            )

            # ======================== BPR损失（辅助损失）========================
            bpr_loss, reg_loss = Recmodel.bpr_loss(batch_users, batch_pos, batch_neg)

            # 联合损失：噪声分类为主，BPR为辅助
            loss = noise_loss + bpr_loss_weight * (bpr_loss + reg_loss)

            aver_bpr_loss += bpr_loss.item()
            aver_noise_loss += noise_loss.item()
        else:
            # 如果不使用噪声标签，回退到原始BPR训练
            bpr_loss, reg_loss = Recmodel.bpr_loss(batch_users, batch_pos, batch_neg)
            loss = bpr_loss + reg_loss
            aver_bpr_loss += bpr_loss.item()

        aver_loss += loss.item()

        # ======================== 反向传播 ========================
        bpr.opt.zero_grad()
        loss.backward()
        bpr.opt.step()

        # ======================== TensorBoard记录 ========================
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', bpr_loss if use_noise_labels else loss,
                         epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            if use_noise_labels:
                w.add_scalar(f'NoiseClassificationLoss/Noise', noise_loss,
                             epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
                w.add_scalar(f'TotalLoss/Total', loss,
                             epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)

    # ======================== 计算平均损失 ========================
    aver_loss = aver_loss / total_batch
    aver_bpr_loss = aver_bpr_loss / total_batch
    aver_noise_loss = aver_noise_loss / total_batch if use_noise_labels else 0.

    time_info = timer.dict()
    timer.zero()

    if use_noise_labels:
        return f"loss{aver_loss:.3f}(noise:{aver_noise_loss:.3f}, bpr:{aver_bpr_loss:.3f})-{time_info}"
    else:
        return f"loss{aver_loss:.3f}-{time_info}"


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}

import torch
import numpy as np
import world
import utils
from typing import Dict, Any


def adversarial_eval_and_train(
        Recmodel,
        loss_class,
        batch_users: torch.Tensor,
        batch_items: torch.Tensor,
        batch_noise_labels: torch.Tensor,
        threshold: float = 0.5,
        use_noise_dropout: bool = True,
) -> Dict[str, Any]:
    """
    在“判别器 + 对抗样本”场景下使用的一步式接口
    """
    device = world.device
    Recmodel = Recmodel.to(device)

    batch_users = batch_users.long().to(device)
    batch_items = batch_items.long().to(device)
    batch_noise_labels = batch_noise_labels.long().to(device)

    if getattr(Recmodel, "noise_mlp", None) is None:
        raise RuntimeError("Recmodel.noise_mlp is None. Please enable use_noise_labels in config.")

    # ======================== 1) 评估阶段（不更新参数） ========================
    Recmodel.eval()
    with torch.no_grad():
        all_users, all_items = Recmodel.computer()
        users_emb_id = all_users[batch_users]
        items_emb_id = all_items[batch_items]

        # =================== 【修改核心 1】拼接语义向量 ===================
        concat_list = [users_emb_id, items_emb_id]

        # 检查是否使用了语义向量
        if hasattr(Recmodel, 'user_semantic') and Recmodel.user_semantic is not None:
            users_emb_sem = Recmodel.user_semantic(batch_users)
            items_emb_sem = Recmodel.item_semantic(batch_items)
            concat_list.append(users_emb_sem)
            concat_list.append(items_emb_sem)

        combined_emb = torch.cat(concat_list, dim=1)
        # =================== 【修改结束】 ===================

        logits = Recmodel.noise_mlp(combined_emb).squeeze(1)
        probs = torch.sigmoid(logits)

        preds = (probs >= threshold).long()
        labels = batch_noise_labels

        correct = (preds == labels).sum().item()
        total = labels.numel()
        acc = correct / total if total > 0 else 0.0

        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        clean_mask = (labels == 0)
        noise_mask = (labels == 1)
        avg_p_clean = probs[clean_mask].mean().item() if clean_mask.any() else 0.0
        avg_p_noise = probs[noise_mask].mean().item() if noise_mask.any() else 0.0

        users_np = batch_users.detach().cpu().numpy()
        items_np = batch_items.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()

        per_sample_results = []
        for u, i, y, y_hat, p in zip(users_np, items_np, labels_np, preds_np, probs_np):
            per_sample_results.append({
                "user_id": int(u),
                "item_id": int(i),
                "true_label": int(y),
                "pred_label": int(y_hat),
                "pred_noise_prob": float(p),
                "is_correct": bool(y == y_hat),
            })

    # ======================== 2) 增量训练阶段（只更新一次） ========================
    Recmodel.train()

    # noise_classification_loss 内部已经处理了语义向量的拼接，所以这里直接调用没问题
    noise_loss = Recmodel.noise_classification_loss(
        users=batch_users,
        items=batch_items,
        noise_labels=batch_noise_labels,
    )

    optimizer = loss_class.opt
    optimizer.zero_grad()
    noise_loss.backward()
    optimizer.step()

    metrics = {
        "accuracy": acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "noise_loss": float(noise_loss.detach().cpu().item()),
        "avg_pred_noise_prob_clean": float(avg_p_clean),
        "avg_pred_noise_prob_noise": float(avg_p_noise),
        "per_sample_results": per_sample_results,
    }

    return metrics


def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)

    # 注意：这里是关键！打印语句必须在 with torch.no_grad() 块之外
    # 且在 pool.close() 之前
    if multicore == 1:
        pool.close()

    # 格式化打印结果 - 放在这里！
    print("\n" + "=" * 90)
    print(f"Epoch [{epoch}] Test Results")
    print("=" * 90)

    # 表头
    header = f"{'Metric':<15}"
    for k in world.topks:
        header += f"@{k:<12}"
    print(header)
    print("-" * 90)

    # Recall 行
    recall_line = f"{'Recall':<15}"
    for i in range(len(world.topks)):
        recall_line += f"{results['recall'][i]:<12.4f}"
    print(recall_line)

    # Precision 行
    precision_line = f"{'Precision':<15}"
    for i in range(len(world.topks)):
        precision_line += f"{results['precision'][i]:<12.4f}"
    print(precision_line)

    # NDCG 行
    ndcg_line = f"{'NDCG':<15}"
    for i in range(len(world.topks)):
        ndcg_line += f"{results['ndcg'][i]:<12.4f}"
    print(ndcg_line)

    print("=" * 90 + "\n")

    return results


# def predict_noise_score(dataset, model, dataset_path, threshold=0.5):
#     """
#     为训练数据中的每个交互打噪声分数
#
#     Args:
#         dataset: 数据集
#         model: 训练好的NoiseDiscriminator模型
#         dataset_path: 数据集路径（清洁数据将保存在该目录下）
#         threshold: 噪声分数阈值 (分数 > threshold 视为噪声)
#
#     Returns:
#         clean_data_dict: {user_id: [clean_items]}
#         noise_stats: 统计信息字典
#     """
#     model.eval()
#     u_batch_size = world.config.get('test_u_batch_size', 256)
#
#     # ======================== 输出文件路径 ========================
#     output_file = os.path.join(dataset_path, 'clean_data.txt')
#
#     # ======================== 存储结果 ========================
#     noise_scores_dict = {}  # {(user, item): noise_score}
#     clean_data_dict = {}  # {user_id: [clean_items]}
#
#     noise_count = 0
#     clean_count = 0
#
#     world.cprint("=" * 80)
#     world.cprint("PREDICTING NOISE SCORES FOR TRAINING DATA")
#     world.cprint("=" * 80)
#
#     with torch.no_grad():
#         # ======================== 遍历所有训练交互 ========================
#         total_interactions = len(dataset.trainUser)
#
#         # 分批处理
#         for batch_start in range(0, total_interactions, u_batch_size):
#             batch_end = min(batch_start + u_batch_size, total_interactions)
#             batch_indices = np.arange(batch_start, batch_end)
#
#             # 获取批处理数据
#             batch_users = torch.LongTensor(dataset.trainUser[batch_indices]).to(world.device)
#             batch_items = torch.LongTensor(dataset.trainItem[batch_indices]).to(world.device)
#
#             # 获取所有用户和物品表征
#             all_users, all_items = model.computer()
#
#             # 获取用户和物品表征
#             users_emb = all_users[batch_users]  # (batch_size, latent_dim)
#             items_emb = all_items[batch_items]  # (batch_size, latent_dim)
#
#             # 拼接表征
#             combined_emb = torch.cat([users_emb, items_emb], dim=1)  # (batch_size, latent_dim*2)
#
#             # MLP预测噪声概率（使用sigmoid转换为概率）
#             noise_logits = model.noise_mlp(combined_emb)  # (batch_size, 1)
#             noise_probs = torch.sigmoid(noise_logits).squeeze(1).cpu().numpy()  # (batch_size,)
#
#             batch_users_np = batch_users.cpu().numpy()
#             batch_items_np = batch_items.cpu().numpy()
#
#             # ======================== 处理批处理结果 ========================
#             for i, (user, item, prob) in enumerate(zip(batch_users_np, batch_items_np, noise_probs)):
#                 noise_scores_dict[(user, item)] = prob
#
#                 # 初始化用户的干净物品列表
#                 if user not in clean_data_dict:
#                     clean_data_dict[user] = []
#
#                 # 根据阈值判断是否为噪声
#                 if prob <= threshold:
#                     clean_data_dict[user].append(item)
#                     clean_count += 1
#                 else:
#                     noise_count += 1
#
#             # 打印进度
#             if (batch_end // u_batch_size) % 10 == 0 or batch_end == total_interactions:
#                 world.cprint(f"Processed {batch_end}/{total_interactions} interactions")
#
#     # ======================== 写入清洁数据文件 ========================
#     world.cprint("=" * 80)
#     world.cprint(f"WRITING CLEAN DATA TO {output_file}")
#     world.cprint("=" * 80)
#
#     os.makedirs(dataset_path, exist_ok=True)
#     with open(output_file, 'w') as f:
#         for user_id in sorted(clean_data_dict.keys()):
#             items = clean_data_dict[user_id]
#             if len(items) > 0:  # 只写入有干净物品的用户
#                 item_str = ' '.join(map(str, items))
#                 f.write(f"{user_id} {item_str}\n")
#
#     world.cprint(f"Clean data written to {output_file}")
#
#     # ======================== 统计信息 ========================
#     total_interactions_original = len(dataset.trainUser)
#     noise_stats = {
#         'total_interactions': total_interactions_original,
#         'clean_interactions': clean_count,
#         'noise_interactions': noise_count,
#         'clean_ratio': clean_count / total_interactions_original,
#         'noise_ratio': noise_count / total_interactions_original,
#         'threshold': threshold,
#         'noise_scores': noise_scores_dict
#     }
#
#     return clean_data_dict, noise_stats

#  加上打印的版本
def predict_noise_score(dataset, model, dataset_path, threshold=0.5):
    """
    为训练数据中的每个交互打噪声分数
    """
    model.eval()
    u_batch_size = world.config.get('test_u_batch_size', 256)

    output_file = os.path.join(dataset_path, 'clean_data.txt')

    noise_scores_dict = {}
    clean_data_dict = {}

    noise_count = 0
    clean_count = 0

    world.cprint("PREDICTING NOISE SCORES FOR TRAINING DATA")

    print("\n" + "=" * 100)
    print(f"{'User':<10} {'Item':<10} {'Noise Prob':<15} {'Decision':<15}")
    print("=" * 100)

    with torch.no_grad():
        total_interactions = len(dataset.trainUser)

        for batch_start in range(0, total_interactions, u_batch_size):
            batch_end = min(batch_start + u_batch_size, total_interactions)
            batch_indices = np.arange(batch_start, batch_end)

            batch_users = torch.LongTensor(
                dataset.trainUser[batch_indices]
            ).to(world.device)
            batch_items = torch.LongTensor(
                dataset.trainItem[batch_indices]
            ).to(world.device)

            all_users, all_items = model.computer()

            users_emb = all_users[batch_users]
            items_emb = all_items[batch_items]

            # =================== 【修改核心 2】拼接语义向量 ===================
            concat_list = [users_emb, items_emb]

            # 检查并拼接语义
            if hasattr(model, 'user_semantic') and model.user_semantic is not None:
                users_emb_sem = model.user_semantic(batch_users)
                items_emb_sem = model.item_semantic(batch_items)
                concat_list.append(users_emb_sem)
                concat_list.append(items_emb_sem)

            combined_emb = torch.cat(concat_list, dim=1)
            # =================== 【修改结束】 ===================

            noise_logits = model.noise_mlp(combined_emb)
            noise_probs = torch.sigmoid(
                noise_logits
            ).squeeze(1).cpu().numpy()

            batch_users_np = batch_users.cpu().numpy()
            batch_items_np = batch_items.cpu().numpy()

            for i, (user, item, prob) in enumerate(
                    zip(batch_users_np, batch_items_np, noise_probs)
            ):
                noise_scores_dict[(user, item)] = prob

                if user not in clean_data_dict:
                    clean_data_dict[user] = []

                decision = "NOISE" if prob > threshold else "CLEAN"
                # 为了防止控制台输出过多刷屏，你可以选择不打印，或者只打印前几个
                # print(f"{user:<10} {item:<10} {prob:<15.6f} {decision:<15}")

                if prob <= threshold:
                    clean_data_dict[user].append(item)
                    clean_count += 1
                else:
                    noise_count += 1

            if (batch_end // u_batch_size) % 10 == 0 or \
                    batch_end == total_interactions:
                world.cprint(
                    f"Processed {batch_end}/{total_interactions} interactions"
                )

    print("=" * 100 + "\n")

    world.cprint(f"WRITING CLEAN DATA TO {output_file}")

    os.makedirs(dataset_path, exist_ok=True)
    with open(output_file, 'w') as f:
        for user_id in sorted(clean_data_dict.keys()):
            items = clean_data_dict[user_id]
            if len(items) > 0:
                item_str = ' '.join(map(str, items))
                f.write(f"{user_id} {item_str}\n")

    world.cprint(f"Clean data written to {output_file}")

    total_interactions_original = len(dataset.trainUser)
    noise_stats = {
        'total_interactions': total_interactions_original,
        'clean_interactions': clean_count,
        'noise_interactions': noise_count,
        'clean_ratio': clean_count / total_interactions_original,
        'noise_ratio': noise_count / total_interactions_original,
        'threshold': threshold,
        'noise_scores': noise_scores_dict
    }

    return clean_data_dict, noise_stats


def plot_noise_score_distribution(noise_stats, save_path=None):
    """
    绘制噪声分数分布直方图

    Args:
        noise_stats: 噪声统计字典
        save_path: 保存图表的路径（可选）
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        world.cprint("matplotlib not installed, skipping plot generation")
        return

    noise_scores = list(noise_stats['noise_scores'].values())
    threshold = noise_stats['threshold']

    plt.figure(figsize=(12, 6))

    # 绘制直方图
    plt.hist(noise_scores, bins=50, alpha=0.7, edgecolor='black')

    # 添加阈值线
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold:.4f}')

    plt.xlabel('Noise Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Noise Scores', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        world.cprint(f"Plot saved to {save_path}")

    plt.close()


def print_noise_statistics(noise_stats):
    """
    打印噪声统计信息
    """
    print("\n" + "=" * 80)
    print("NOISE FILTERING STATISTICS")
    print("=" * 80)
    print(f"{'Total interactions':<30}: {noise_stats['total_interactions']}")
    print(f"{'Clean interactions':<30}: {noise_stats['clean_interactions']}")
    print(f"{'Noise interactions':<30}: {noise_stats['noise_interactions']}")
    print(f"{'Clean ratio':<30}: {noise_stats['clean_ratio']:.2%}")
    print(f"{'Noise ratio':<30}: {noise_stats['noise_ratio']:.2%}")
    print(f"{'Threshold':<30}: {noise_stats['threshold']:.4f}")
    print("=" * 80 + "\n")

