"""
Integrated MF (Matrix Factorization) Training Script
Combines dataset loading, model definition, training, and parameter saving
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
from tqdm import tqdm
import json


# ======================== CONFIG ========================

class Config:
    """配置类"""

    def __init__(self, dataset_name='ml1m'):
        # 基础配置
        self.dataset = dataset_name  # 'ml1m', 'book-crossing', 'dbbook2014'

        # 获取项目根目录
        current_dir = Path(__file__).parent.parent.parent  # recommender/models/train_mf.py -> 根目录
        self.data_path = current_dir / 'data' / dataset_name
        self.model_save_path = current_dir / 'recommenders' / 'weights'

        # 验证数据路径
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")

        print(f"✓ Data path: {self.data_path}")
        print(f"✓ Model save path: {self.model_save_path}")

        # 模型配置
        self.latent_dim_rec = 64

        # 训练配置
        self.batch_size = 2048
        self.test_batch_size = 512
        self.lr = 0.001
        self.decay = 1e-5
        self.epochs = 100
        self.seed = 2020

        # 其他
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.topks = [10, 20, 50]
        self.tensorboard = False


# ======================== DATASET ========================

class BasicDataset:
    """基础数据集类"""

    def __init__(self):
        print("Initializing dataset...")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError


class Loader(BasicDataset):
    """数据加载类"""

    def __init__(self, config):
        print(f'Loading dataset: {config.dataset}')

        # 直接使用 config 中的数据路径
        data_path = str(config.data_path)

        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']

        self.n_user = 0
        self.m_item = 0

        train_file = os.path.join(data_path, 'train.txt')
        test_file = os.path.join(data_path, 'test.txt')
        self.path = data_path

        # 验证文件存在
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Train file not found: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")

        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []

        self.traindataSize = 0
        self.testDataSize = 0

        # 加载训练集
        print("Loading training data...")
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)

        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        # 加载测试集
        print("Loading test data...")
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)

        self.m_item += 1
        self.n_user += 1

        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        print(f"✓ Training interactions: {self.traindataSize}")
        print(f"✓ Test interactions: {self.testDataSize}")
        print(f"✓ Users: {self.n_user}, Items: {self.m_item}")
        print(f"✓ Sparsity: {(self.traindataSize + self.testDataSize) / self.n_user / self.m_item:.4f}")

        # 构建用户-物品交互矩阵
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item)
        )

        # 预计算所有正样本
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()

        print("✓ Dataset loaded successfully\n")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def __build_test(self):
        """构建测试字典"""
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if user not in test_data:
                test_data[user] = []
            test_data[user].append(item)
        return test_data

    def getUserPosItems(self, users):
        """获取用户的正样本物品"""
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserItemFeedback(self, users, items):
        """获取用户对物品的反馈"""
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))


# ======================== MODEL ========================

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PureMF(BasicModel):
    """Matrix Factorization 模型"""

    def __init__(self, config, dataset):
        super(PureMF, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = config.device

        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config.latent_dim_rec

        # 初始化嵌入
        self.__init_weight()
        self.f = nn.Sigmoid()

        print(f"✓ PureMF model initialized")
        print(f"  - Users: {self.num_users}, Items: {self.num_items}")
        print(f"  - Latent dim: {self.latent_dim}")

    def __init_weight(self):
        """初始化权重"""
        self.embedding_user = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # 使用正态分布初始化
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        print("✓ Using Normal distribution N(0, 0.1) initialization for PureMF")

    def getUsersRating(self, users):
        """获取用户对所有物品的评分"""
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        """BPR 损失"""
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))

        reg_loss = (1 / 2) * (
                users_emb.norm(2).pow(2) +
                pos_emb.norm(2).pow(2) +
                neg_emb.norm(2).pow(2)
        ) / float(len(users))

        return loss, reg_loss

    def forward(self, users, items):
        """前向传播"""
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)

    def predict(self, user_ids, item_ids=None):
        """
        用于生成排名

        Args:
            user_ids: 用户ID列表或数组，shape (batch_size,)
            item_ids: 如果为 None，预测所有物品的评分

        Returns:
            scores: numpy 数组，shape (batch_size, n_items) 或 (batch_size,)
        """
        self.eval()

        with torch.no_grad():
            # 转换为 tensor
            if not isinstance(user_ids, torch.Tensor):
                user_ids = torch.LongTensor(user_ids)

            user_ids = user_ids.to(self.device)

            # 如果 item_ids 为 None，预测所有物品
            if item_ids is None:
                rating = self.getUsersRating(user_ids)
                return rating.cpu().numpy()
            else:
                # 预测特定物品的评分
                if not isinstance(item_ids, torch.Tensor):
                    item_ids = torch.LongTensor(item_ids)

                item_ids = item_ids.to(self.device)
                scores = self.forward(user_ids, item_ids)

                return scores.cpu().numpy()


# ======================== TRAINING ========================

class BPRLoss:
    """BPR 损失类"""

    def __init__(self, model, config):
        self.model = model
        self.weight_decay = config.decay
        self.lr = config.lr
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        """单步训练"""
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        total_loss = loss + reg_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.cpu().item()


class Trainer:
    """训练类"""

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.device = config.device

        # 初始化模型
        self.model = PureMF(config, dataset).to(self.device)
        self.loss_fn = BPRLoss(self.model, config)

        # 创建保存目录 - 使用 config 中的数据集名称
        self.model_save_dir = Path(config.model_save_path) / config.dataset / 'MF' / 'best_model'
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Model will be saved to: {self.model_save_dir}\n")

        self.best_recall = 0.0
        self.patience = 10
        self.patience_count = 0

    def sample_negative(self, batch_size=None):
        """采样负样本"""
        if batch_size is None:
            batch_size = self.config.batch_size

        dataset = self.dataset
        allPos = dataset.allPos
        n_users = dataset.n_users
        n_items = dataset.m_items

        users = np.random.randint(0, n_users, batch_size)
        S = []

        for user in users:
            posForUser = allPos[user]
            if len(posForUser) == 0:
                continue

            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]

            # 采样负样本
            while True:
                negitem = np.random.randint(0, n_items)
                if negitem not in posForUser:
                    break

            S.append([user, positem, negitem])

        return np.array(S)

    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()

        total_loss = 0.0
        steps = 0

        # 采样训练数据
        S = self.sample_negative()
        users = torch.LongTensor(S[:, 0]).to(self.device)
        pos_items = torch.LongTensor(S[:, 1]).to(self.device)
        neg_items = torch.LongTensor(S[:, 2]).to(self.device)

        # 分批训练
        n_batches = len(users) // self.config.batch_size + 1

        pbar = tqdm(total=n_batches, desc=f"Epoch {epoch + 1}")

        for i in range(0, len(users), self.config.batch_size):
            end_idx = min(i + self.config.batch_size, len(users))

            batch_users = users[i:end_idx]
            batch_pos = pos_items[i:end_idx]
            batch_neg = neg_items[i:end_idx]

            loss = self.loss_fn.stageOne(batch_users, batch_pos, batch_neg)
            total_loss += loss
            steps += 1

            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        pbar.close()

        avg_loss = total_loss / steps
        return avg_loss

    def evaluate(self):
        """评估模型"""
        self.model.eval()

        testDict = self.dataset.testDict
        dataset = self.dataset

        results = {'recall@10': 0.0, 'recall@20': 0.0, 'recall@50': 0.0,
                   'ndcg@10': 0.0, 'ndcg@20': 0.0, 'ndcg@50': 0.0}

        all_recall = []
        all_ndcg = []

        with torch.no_grad():
            users = list(testDict.keys())

            # 分批评估
            for batch_idx in range(0, len(users), self.config.test_batch_size):
                end_idx = min(batch_idx + self.config.test_batch_size, len(users))
                batch_users = users[batch_idx:end_idx]

                # 获取排名
                batch_users_tensor = torch.LongTensor(batch_users).to(self.device)
                rating = self.model.getUsersRating(batch_users_tensor)

                # 排除已交互的物品
                allPos = dataset.getUserPosItems(batch_users)
                for idx, pos_items in enumerate(allPos):
                    rating[idx, pos_items] = -999999

                # 获取 top-50
                _, top_50 = torch.topk(rating, k=50, dim=1)
                top_50 = top_50.cpu().numpy()

                # 计算指标
                for idx, user in enumerate(batch_users):
                    ground_truth = testDict[user]
                    pred_items = top_50[idx]

                    # Recall@10
                    recall_10 = len(set(pred_items[:10]) & set(ground_truth)) / len(ground_truth)
                    # Recall@20
                    recall_20 = len(set(pred_items[:20]) & set(ground_truth)) / len(ground_truth)
                    # Recall@50
                    recall_50 = len(set(pred_items[:50]) & set(ground_truth)) / len(ground_truth)

                    # NDCG@10
                    dcg_10 = 0.0
                    for i, item in enumerate(pred_items[:10]):
                        if item in ground_truth:
                            dcg_10 += 1.0 / np.log2(i + 2)
                    idcg_10 = sum([1.0 / np.log2(i + 2) for i in range(min(10, len(ground_truth)))])
                    ndcg_10 = dcg_10 / idcg_10 if idcg_10 > 0 else 0

                    # NDCG@20
                    dcg_20 = 0.0
                    for i, item in enumerate(pred_items[:20]):
                        if item in ground_truth:
                            dcg_20 += 1.0 / np.log2(i + 2)
                    idcg_20 = sum([1.0 / np.log2(i + 2) for i in range(min(20, len(ground_truth)))])
                    ndcg_20 = dcg_20 / idcg_20 if idcg_20 > 0 else 0

                    # NDCG@50
                    dcg_50 = 0.0
                    for i, item in enumerate(pred_items[:50]):
                        if item in ground_truth:
                            dcg_50 += 1.0 / np.log2(i + 2)
                    idcg_50 = sum([1.0 / np.log2(i + 2) for i in range(min(50, len(ground_truth)))])
                    ndcg_50 = dcg_50 / idcg_50 if idcg_50 > 0 else 0

                    all_recall.append([recall_10, recall_20, recall_50])
                    all_ndcg.append([ndcg_10, ndcg_20, ndcg_50])

        all_recall = np.array(all_recall).mean(axis=0)
        all_ndcg = np.array(all_ndcg).mean(axis=0)

        results['recall@10'] = all_recall[0]
        results['recall@20'] = all_recall[1]
        results['recall@50'] = all_recall[2]
        results['ndcg@10'] = all_ndcg[0]
        results['ndcg@20'] = all_ndcg[1]
        results['ndcg@50'] = all_ndcg[2]

        return results

    def save_model(self, epoch):
        """保存模型"""
        # 保存模型权重
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.loss_fn.optimizer.state_dict(),
        }

        checkpoint_file = self.model_save_dir / f'epoch={epoch}.checkpoint.pth.tar'
        torch.save(checkpoint, checkpoint_file)
        print(f"✓ Model checkpoint saved: {checkpoint_file}")

        # 保存模型参数
        args_dict = {
            'modeltype': 'PureMF',
            'dataset': self.config.dataset,
            'latent_dim_rec': self.config.latent_dim_rec,
            'lr': self.config.lr,
            'decay': self.config.decay,
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs,
        }

        args_file = self.model_save_dir / 'args.txt'
        with open(args_file, 'w') as f:
            json.dump(args_dict, f, indent=2)
        print(f"✓ Model args saved: {args_file}")

    def train(self):
        """主训练循环"""
        print("\n" + "=" * 80)
        print("Starting PureMF Training")
        print("=" * 80 + "\n")

        for epoch in range(self.config.epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch + 1}/{self.config.epochs} | Loss: {train_loss:.4f}")

            # 评估
            if (epoch + 1) % 10 == 0:
                print("\nEvaluating...")
                results = self.evaluate()

                print(f"\n{'=' * 80}")
                print(f"Evaluation Results - Epoch {epoch + 1}")
                print(f"{'=' * 80}")
                print(f"Recall@10:  {results['recall@10']:.4f}")
                print(f"Recall@20:  {results['recall@20']:.4f}")
                print(f"Recall@50:  {results['recall@50']:.4f}")
                print(f"NDCG@10:    {results['ndcg@10']:.4f}")
                print(f"NDCG@20:    {results['ndcg@20']:.4f}")
                print(f"NDCG@50:    {results['ndcg@50']:.4f}")
                print(f"{'=' * 80}\n")

                # 保存最好的模型
                if results['recall@20'] > self.best_recall:
                    self.best_recall = results['recall@20']
                    self.patience_count = 0
                    self.save_model(epoch)
                    print(f"✓ Best model updated (Recall@20: {self.best_recall:.4f})\n")
                else:
                    self.patience_count += 1
                    print(f"No improvement for {self.patience_count} evaluations\n")

                # 早停
                if self.patience_count >= self.patience:
                    print(f"✗ Early stopping after {self.patience} evaluations without improvement")
                    break

        print("\n" + "=" * 80)
        print("Training Completed!")
        print(f"Best model saved to: {self.model_save_dir}")
        print("=" * 80 + "\n")


# ======================== MAIN ========================

def main(dataset_name='ml1m'):
    """主函数"""
    # 支持的数据集列表
    supported_datasets = ['ml1m', 'book-crossing', 'dbbook2014']

    if dataset_name not in supported_datasets:
        print(f"❌ Unsupported dataset: {dataset_name}")
        print(f"Supported: {supported_datasets}")
        return

    # 设置随机种子
    np.random.seed(2020)
    torch.manual_seed(2020)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2020)

    # 初始化配置
    config = Config(dataset_name=dataset_name)

    print(f"\n{'=' * 80}")
    print("PureMF Configuration")
    print(f"{'=' * 80}")
    print(f"Dataset: {config.dataset}")
    print(f"Device: {config.device}")
    print(f"Latent Dim: {config.latent_dim_rec}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.lr}")
    print(f"Epochs: {config.epochs}")
    print(f"{'=' * 80}\n")

    # 加载数据集
    dataset = Loader(config)

    # 创建训练器
    trainer = Trainer(config, dataset)

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train PureMF model')
    parser.add_argument('--dataset', type=str, default='dbbook2014',
                        choices=['ml1m', 'book-crossing', 'dbbook2014'],
                        help='Dataset name')

    args = parser.parse_args()
    main(dataset_name=args.dataset)
