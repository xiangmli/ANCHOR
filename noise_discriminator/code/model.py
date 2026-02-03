import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import os
current_dir = os.path.dirname(os.path.abspath(__file__))


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)


class NoiseDiscriminator(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(NoiseDiscriminator, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        # ======================== 1. 加载语义向量 ========================
        self.semantic_dim = 0  # 初始为0，如果加载成功则更新
        self.user_semantic = None
        self.item_semantic = None

        # 假设语义文件在 './pretrained_emb' 目录下
        dataset_name = self.config['dataset']  # 确保 config 中有 dataset 名字
        #emb_dir = './pretrained_emb'
        emb_dir = os.path.join(current_dir, 'pretrained_emb')
        user_emb_path = os.path.join(emb_dir, f'{dataset_name}_user_semantic_emb.pt')
        item_emb_path = os.path.join(emb_dir, f'{dataset_name}_item_semantic_emb.pt')

        if self.config['use_semantic']:  # 在 config 中增加一个开关
            if os.path.exists(user_emb_path) and os.path.exists(item_emb_path):
                # 加载并冻结参数（不需要梯度更新）
                self.user_semantic = nn.Embedding.from_pretrained(
                    torch.load(user_emb_path).float(), freeze=True)
                self.item_semantic = nn.Embedding.from_pretrained(
                    torch.load(item_emb_path).float(), freeze=True)

                self.semantic_dim = self.user_semantic.embedding_dim  # 通常是 384
                world.cprint(f"Loaded Semantic Embeddings! Dim: {self.semantic_dim}")
            else:
                world.cprint(f"Warning: Semantic files not found at {user_emb_path} or {item_emb_path}")

        # ======================== 2. ID 嵌入层 (原有逻辑) ========================
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

        # ======================== 3. 构建 MLP ========================
        # 注意：现在的输入维度变了
        if self.config['use_noise_labels']:
            self.noise_mlp = self._build_noise_mlp()
            world.cprint('Noise classification MLP initialized')
        else:
            self.noise_mlp = None

        print(f"lgn is already to go(dropout:{self.config['dropout']})")

    def _build_noise_mlp(self):
        """
        构建噪声分类MLP
        输入：(用户ID表征 + 物品ID表征) + (用户语义表征 + 物品语义表征)
        """
        mlp_layers = self.config.get('noise_mlp_layers', [256, 128, 1])
        mlp_dropout = self.config.get('noise_mlp_dropout', 0.1)

        # ID embedding (latent_dim * 2) + Semantic embedding (semantic_dim * 2)
        input_dim = (self.latent_dim * 2) + (self.semantic_dim * 2)

        layers = []
        for i, hidden_dim in enumerate(mlp_layers[:-1]):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if mlp_dropout > 0:
                layers.append(nn.Dropout(mlp_dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, mlp_layers[-1]))
        return nn.Sequential(*layers)

    # ... (保留 __dropout_x, __dropout, computer 方法不变) ...
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss, reg_loss

    # ======================== 4. 修改噪声分类损失 ========================
    def noise_classification_loss(self, users, items, noise_labels):
        all_users, all_items = self.computer()

        # ID Embeddings (来自 GCN)
        users_emb_id = all_users[users.long()]
        items_emb_id = all_items[items.long()]

        # 对 ID Embedding 进行 Dropout (原有逻辑)
        noise_dropout = self.config.get('noise_user_embedding_dropout', 0.5)
        if self.training and noise_dropout > 0:
            users_emb_id = torch.nn.functional.dropout(users_emb_id, p=noise_dropout, training=True)

        # 准备拼接列表
        concat_list = [users_emb_id, items_emb_id]

        # 拼接 Semantic Embeddings (如果有)
        if self.user_semantic is not None and self.item_semantic is not None:
            # 查找语义向量 (注意: 这里的 users/items 是 batch index)
            # 使用 .to(device) 确保在同一个设备上，虽然 nn.Embedding 应该已经处理好了
            users_emb_sem = self.user_semantic(users.long())
            items_emb_sem = self.item_semantic(items.long())

            # 你也可以选择在这里对语义向量加 dropout，防止过拟合
            # if self.training:
            #     users_emb_sem = torch.nn.functional.dropout(users_emb_sem, p=0.1, training=True)

            concat_list.append(users_emb_sem)
            concat_list.append(items_emb_sem)

        # 最终拼接：[User_ID, Item_ID, User_Sem, Item_Sem]
        combined_emb = torch.cat(concat_list, dim=1)

        # MLP 预测
        noise_logits = self.noise_mlp(combined_emb)
        noise_logits = noise_logits.squeeze(1)

        loss_fn = nn.BCEWithLogitsLoss()
        noise_labels_float = noise_labels.float()
        loss = loss_fn(noise_logits, noise_labels_float)

        return loss

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


